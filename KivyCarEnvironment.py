import multiprocessing
import time
from map import CarApp
import numpy as np



def worker(start_event, reset_q, mode_q, state_q, action_q, next_state_reward_done_tuple_q):
    car_app_instance = CarApp(start_event, reset_q, mode_q, state_q, action_q, next_state_reward_done_tuple_q)
    car_app_instance.run()
	
class KivyCarEnvironment(object):

    def __init__(self):
        self.start_event = multiprocessing.Event()
        self.reset_q = multiprocessing.Queue()
        self.mode_q = multiprocessing.Queue()
        self.state_q = multiprocessing.Queue()
        self.action_q = multiprocessing.Queue()
        self.next_state_reward_done_tuple_q = multiprocessing.Queue()
        self.process = None
		
    def start(self):
        self.process = multiprocessing.Process(target=worker, args=(self.start_event, self.reset_q, self.mode_q, self.state_q, self.action_q, self.next_state_reward_done_tuple_q))
        self.process.start()
        time.sleep(10)
		
    def close(self):
        if self.process is not None:
            self.process.join()
			
    def reset(self, mode="Train"):
        self.reset_q.put(True)
        self.mode_q.put(mode)
        self.start_event.set()
        return self.state_q.get()

	
    def step(self, action):
        self.action_q.put(action)
        return self.next_state_reward_done_tuple_q.get()
		
    def action_space_sample(self):
	    return np.random.uniform(-5, 5, 2)
		
    def action_space_shape(self):
        return 1
		
    def action_space_low(self):
        return -5.0
	
    def action_space_high(self):
        return 5.0

    def max_episode_steps(self):
        return 2500



if __name__ == '__main__':
    env = KivyCarEnvironment()
    env.start()
    state = env.reset()
    print("Main: Got state: ", state)
    done = False
	
    while not done:
        #action = np.random.randint(3)
        action = env.action_space_sample()
        obs, reward, done, _  = env.step(action)
        print("reward: ", reward, ", done: ", done, ", obs", obs)
		
    state = env.reset("Eval")
    print("Main: Got state: ", state)
    done = False
	
    while not done:
        #action = np.random.randint(3)
        action = env.action_space_sample()
        obs, reward, done, _  = env.step(action)
        print("reward: ", reward, ", done: ", done, ", obs", obs)
  
    env.close()
