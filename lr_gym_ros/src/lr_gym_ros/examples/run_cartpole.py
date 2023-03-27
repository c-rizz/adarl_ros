#!/usr/bin/env python3

from lr_gym.envs.CartpoleEnv import CartpoleEnv
from lr_gym.envs.GymEnvWrapper import GymEnvWrapper
from lr_gym_ros.envControllers.GazeboController import GazeboController
import lr_gym.utils.utils
import time
import numpy as np
import lr_gym.utils.dbg.ggLog as ggLog
from typing import List, Tuple, Callable, Dict, Union, Any
import tqdm



def evaluatePolicy(env, model, episodes : int, on_ep_done_callback = None, predict_func : Callable[[Any], Tuple[Any,Any]] = None, progress_bar : bool = True, images_return = None, obs_return = None):
    if predict_func is None:
        predict_func = model.predict
    rewards = np.empty((episodes,), dtype = np.float32)
    steps = np.empty((episodes,), dtype = np.int32)
    wallDurations = np.empty((episodes,), dtype = np.float32)
    predictWallDurations = np.empty((episodes,), dtype = np.float32)
    totDuration=0.0
    successes = 0.0
    #frames = []
    #do an average over a bunch of episodes
    if not progress_bar:
        maybe_tqdm = lambda x:x
    else:
        maybe_tqdm = tqdm.tqdm
    for episode in maybe_tqdm(range(0,episodes)):
        frame = 0
        episodeReward = 0
        done = False
        predDurations = []
        t0 = time.monotonic()
        # ggLog.info("Env resetting...")
        obs = env.reset()
        # ggLog.info("Env resetted")
        if images_return is not None:
            images_return.append([])
        if obs_return is not None:
            obs_return.append([])
        while not done:
            t0_pred = time.monotonic()
            # ggLog.info("Predicting")
            if images_return is not None:
                images_return[-1].append(env.render())
            if obs_return is not None:
                obs_return[-1].append(obs)
            action, _states = predict_func(obs)
            predDurations.append(time.monotonic()-t0_pred)
            # ggLog.info("Stepping")
            obs, stepReward, done, info = env.step(action)
            frame+=1
            episodeReward += stepReward
            # ggLog.info(f"Step reward = {stepReward}")
        rewards[episode]=episodeReward
        if "success" in info.keys():
            if info["success"]:
                ggLog.info(f"Success {successes} ratio = {successes/(episode+1)}")
                successes += 1
        steps[episode]=frame
        wallDurations[episode]=time.monotonic() - t0
        predictWallDurations[episode]=sum(predDurations)
        if on_ep_done_callback is not None:
            on_ep_done_callback(episodeReward=episodeReward, steps=frame, episode=episode)
        ggLog.debug("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
    eval_results = {"reward_mean" : np.mean(rewards),
                    "reward_std" : np.std(rewards),
                    "steps_mean" : np.mean(steps),
                    "steps_std" : np.std(steps),
                    "success_ratio" : successes/episodes,
                    "wall_duration_mean" : np.mean(wallDurations),
                    "wall_duration_std" : np.std(wallDurations),
                    "predict_wall_duration_mean" : np.mean(predictWallDurations),
                    "predict_wall_duration_std" : np.std(predictWallDurations)}
    return eval_results


def main():
    render = True
    stepLength_sec = 0.05
    env = GymEnvWrapper(CartpoleEnv(startSimulation=True,
                                    environmentController = GazeboController(stepLength_sec=stepLength_sec),
                                    render=render))

    images = [] if render else None
    obs = []
    t0 = time.monotonic()
    def policy(obs):
        return 1 if obs[3] > 0 else 0, 0
    res = evaluatePolicy(env = env, model = None, episodes = 10, predict_func=policy,
                                            images_return = images, obs_return=obs)
    t1 = time.monotonic()

    eps = len(obs)
    steps = sum([len(i) for i in obs])
    print(f"Summary:\n{res}")
    print(f"took {t1-t0}s, {steps/(t1-t0)} fps, sim/real = {stepLength_sec*steps/(t1-t0):.2f}x")
    print(f"took {t1-t0}s, {steps/(t1-t0)} fps")
    if images is not None:
        print(f"images[0][0].shape = {images[0][0].shape}")
    newline = "\n"
    print(f"final obss = {newline.join([str(ep[-1]) for ep in obs])}")


if __name__ == "__main__":
    main()