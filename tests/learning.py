
import os
import torch
import datetime
import ray.rllib.agents.a3c as ac
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from ray.tune.registry import register_env
from task_network_env.module.envs import *

class Policy(TorchModelV2, torch.nn.Module):
    def __init__(self, obsSpace, actionSpace, numOutputs, modelConfig, name, nodes) -> None:
        torch.nn.Module.__init__(self)
        TorchModelV2.__init__(self, obsSpace, actionSpace, numOutputs, modelConfig, name)
        self._seed = 0
        self.nodes = int(nodes)

        # selection layer
        selectionOutChannels = 1
        selectionRows = 2
        selectionFeatures = selectionOutChannels * self.nodes
        self._selectionLayer = SlimConv2d(
            in_channels=1,
            out_channels=selectionOutChannels,
            kernel=(selectionRows,1),
            stride=1,
            padding=0,
            activation_fn=torch.nn.Tanh,
            initializer=normc_initializer(1.0),
        )

        # scale layer
        scaleFeatures = 3
        self._scaleLayer = SlimFC(
            in_size=scaleFeatures,
            out_size=scaleFeatures,
            initializer=normc_initializer(1.0),
            activation_fn=torch.nn.Tanh,
        )

        # combine layer
        combinedFeatures = scaleFeatures + selectionFeatures
        self._combinedLayer = SlimFC(
            in_size=combinedFeatures,
            out_size=combinedFeatures,
            initializer=normc_initializer(1.0),
            activation_fn=torch.nn.Tanh,
        )

        # logit layer
        self.actor = SlimFC(
            in_size=combinedFeatures,
            out_size=2 + 2 * self.nodes,
            initializer=normc_initializer(0.01),
            activation_fn=torch.nn.Tanh,
        )

        # value layer
        self._hidden = None
        self.critic = SlimFC(
            in_size=combinedFeatures,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None,
        )

    def selectionLayer(self, inputDict):

        # obs
        taskArray = inputDict["obs"]["availableTaskArray"]
        descendantArray = inputDict["obs"]["availableDescendantArray"]

        # arrange
        input = torch.cat([
            taskArray.unsqueeze(2),
            descendantArray.unsqueeze(2),
        ], dim=2)
        input = input.unsqueeze(1).transpose(2,3)

        # forward selection layer
        input = self._selectionLayer(input)

        # rearrange
        input = input.squeeze(2).transpose(1,2)
        input = torch.flatten(input, start_dim=1)
        return input

    def scaleLayer(self, inputDict):
        
        # obs
        concurrentTasks = inputDict["obs"]["concurrentTasks"]
        availableTasks = inputDict["obs"]["availableTasks"]

        # arrange
        input = torch.cat([
            concurrentTasks,
            availableTasks,
            availableTasks - concurrentTasks,
        ], dim=1)

        # forward scale layer
        input = self._scaleLayer(input)
        return input

    def combinedLayer(self, scaleInput, selectionInput):

        # arange
        input = torch.cat([
            scaleInput,
            selectionInput
        ], dim=1)

        # forward
        input = self._combinedLayer(input)
        return input

    def forward(self, inputDict, state, seqLens):
        selectionInput = self.selectionLayer(inputDict)
        scaleInput = self.scaleLayer(inputDict)
        self._hidden = self.combinedLayer(scaleInput, selectionInput)
        logit = self.actor(self._hidden)
        return logit, state

    def value_function(self):
        return torch.reshape(self.critic(self._hidden), [-1])

def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["concurrentTasks"] = []
    episode.hist_data["concurrentTasks"] = []
    episode.user_data["availableTasks"] = []
    episode.hist_data["availableTasks"] = []

def on_episode_step(info):
    episode = info["episode"]
    lastObs = episode.last_observation_for()
    lastObsRaw = episode.last_raw_obs_for()

    # concurrentTasks
    concurrentTasks = lastObs['concurrentTasks'][0]
    assert concurrentTasks == lastObsRaw['concurrentTasks'][0]

    # availableTasks
    availableTasks = lastObs['availableTasks'][0]
    assert availableTasks == lastObsRaw['availableTasks'][0]

    episode.user_data["concurrentTasks"].append(concurrentTasks)
    episode.user_data["availableTasks"].append(availableTasks)

def on_episode_end(info):
    episode = info["episode"]
    concurrentTasks = np.mean(episode.user_data["concurrentTasks"])
    availableTasks = np.mean(episode.user_data["availableTasks"])

    episode.custom_metrics["concurrentTasks"] = concurrentTasks
    episode.custom_metrics["availableTasks"] = availableTasks

    episode.hist_data["concurrentTasks"] = episode.user_data["concurrentTasks"]
    episode.hist_data["availableTasks"] = episode.user_data["availableTasks"]

def testRun(
    jsonFilePath,
    csvFilePath,
    nodes,
    taskSize,
    workers,
    discountRate,
    episodes,
):

    # register
    ModelCatalog.register_custom_model('policy', Policy)
    env = ContinuousFileTaskNetworkEnvRllib
    register_env('env', lambda envConfig : env(envConfig))

    # config
    # https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#a3c
    config = ac.DEFAULT_CONFIG.copy()
    config.update({
        "framework": 'torch',
        "log_level": 'WARN',

        # trainer
        "rollout_fragment_length": nodes,
        "train_batch_size" : nodes,
        "num_workers": workers,
        "seed": 23,
        "horizon" : nodes,

        # env
        "env": 'env',
        "env_config": {
            "csvFilePath": csvFilePath,
            "jsonFilePath": jsonFilePath,
            "nodes" : nodes,
            "taskSize" : taskSize,
        },

        # model
        "model": {
            "custom_model": 'policy',
            "custom_model_config": {
                "nodes" : nodes,
            },
        },
        "use_gae": False,
        "gamma": discountRate,
        "preprocessor_pref" : None,
        "_disable_preprocessor_api": True, # NOTE if not disabled preprocessing can screw up the observation based on its size

        # save metrics
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
        },

        # evaluate
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_duration": 1,
        "evaluation_duration_unit": "episodes",
        "evaluation_config": {
            "env_config": {
                "csvFilePath": csvFilePath,
                "jsonFilePath": jsonFilePath,
                "nodes" : nodes,
                "taskSize" : taskSize,
            },
            "model": {
                "custom_model": 'policy',
                "custom_model_config": {
                    "nodes" : nodes,
                },
            },
        },
    })
    trainer = ac.A3CTrainer(config=config)
    
    # train
    fileName = f"output_{str(datetime.datetime.now()).split('.')[0].replace(':','.')}.csv"
    filePath = os.path.join(os.path.dirname(__file__), fileName)
    types = ["concurrentTasks", "availableTasks"]
    with open(filePath, 'a') as csvFile:
        columns = ['epoch', 'episodelength', 'reward', 'domain', 'type'] + ["step"+str(i) for i in range(1, nodes + 1)]
        csvFile.write(','.join(columns))

        decimals = 2
        for episode in range(episodes):
            results = trainer.train()
            trainRewardMean = results['episode_reward_mean']
            trainLenMean = results['episode_len_mean']
            
            evalRewardMean = results['evaluation']['episode_reward_mean']
            evalLenMean = results['evaluation']['episode_len_mean']
            print(f"Episode: {episode + 1}\n   train mean reward={np.round(trainRewardMean,decimals)}, len={np.round(trainLenMean,decimals)}\n   eval  mean reward={np.round(evalRewardMean,decimals)}, len={np.round(evalLenMean,decimals)}", flush=True)

            # train
            i0 = 0
            l = len(results["hist_stats"]["episode_lengths"])
            means = {type : np.zeros(nodes) for type in types}
            for el in results["hist_stats"]["episode_lengths"]:
                el = min([nodes, el])
                i1 = i0 + el
                padding = [0]*(nodes - el)
                for type in types:
                    data = results["hist_stats"][type][i0:i1] + padding
                    assert len(data) == nodes
                    means[type] += np.array(data)
                i0 = i1
            for type in types:
                means[type] /= l
                means[type][means[type] == 0] = np.nan
                csvFile.write(f"\n{episode},{trainLenMean},{trainRewardMean},train,{type}," + ','.join([str(x) for x in means[type]]))
            
            # eval
            i0 = 0
            l = len(results["evaluation"]["hist_stats"]["episode_lengths"])
            means = {type : np.zeros(nodes) for type in types}
            for el in results["evaluation"]["hist_stats"]["episode_lengths"]:
                el = min([nodes, el])
                i1 = i0 + el
                padding = [0]*(nodes - el)
                for type in types:
                    data = results["evaluation"]["hist_stats"][type][i0:i1] + padding
                    assert len(data) == nodes
                    means[type] += np.array(data)
                i0 = i1
            for type in types:
                means[type] /= l
                means[type][means[type] == 0] = np.nan
                csvFile.write(f"\n{episode},{trainLenMean},{trainRewardMean},evaluation,{type}," + ','.join([str(x) for x in means[type]]))

if __name__ == "__main__":

    # nodes = 64
    nodes = 209

    taskSize = None
    # taskSize = 5

    discountRate = 1

    workers = 1
    
    episodes = 1000
    
    csvFilePath = r"/usr/src/tests/Duplex_A_20110907.csv"
    jsonFilePath = r"/usr/src/tests/Duplex_A_2011090710-2500.json"

    # csvFilePath = None
    # jsonFilePath = None
    
    testRun(
        jsonFilePath,
        csvFilePath,
        nodes,
        taskSize,
        workers,
        discountRate,
        episodes,
    )
