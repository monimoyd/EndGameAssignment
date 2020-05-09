# Self Driving Car

# Importing the libraries
import torch
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import os

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, BoundedNumericProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.properties import ListProperty
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

np.random.seed(41)


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0



last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
first_update = True
max_steps = 2500
current_step = 0
done = True
on_road = -1
on_road_count = 0
off_road_count = 0
boundary_hit_count = 0
goal_hit_count = 0
train_episode_num = 0
eval_episode_num = 0
mode="Eval"
if os.path.exists("results") == False:
    os.makedirs("results")

if os.path.exists("sand_images") == False:
    os.makedirs("sand_images")	
	
	
full_eval_on_road_stats_file= open("results/full_eval_on_road_stats.csv","w")
eval_on_road_stats_file= open("results/eval_on_road_stats.csv","a")
train_on_road_stats_file= open("results/train_on_road_stats.csv","a")
full_eval_traversal_log= open("results/full_eval_traversal_log.txt","w")
eval_traversal_log= open("results/eval_traversal_log.txt","w")
train_traversal_log= open("results/train_traversal_log.txt","w")
traversal_log = None

img = None
car_img = None
global_counter = 0
digit_images = []
episode_total_reward = 0.0

# This flag full_eval_demo_mode should be enabled only for demo in Full_Eval mode.
# If you want random on road location, change the full_eval_demo_mode to False
full_eval_demo_mode = True


#on_road_postions=[(1247,623), (750,360),(1220,300),(360,310)] # Avg Score: 815.70 episode:148
#on_road_postions=[(1000,590), (750,360),(1220,300),(360,310)] # Avg Score: 1002 episode:148
#on_road_postions=[(1070,490), (580,350)] # Avg Score: 855, episode:148/150 AvG:890 episode:147 Avg:42 episode:120
#on_road_postions=[(1031,496), (343,189)] # Avg Score: 261 episode: 147
#on_road_postions=[(1031,496), (580,350)] # Avg Score: 261 episode: 147

random_location=True



def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global done
    global max_steps
    global current_step
    global img
    global car_img
    global global_counter
    global digit_images
	
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    car_img = PILImage.open("./images/latest_triangle_car.png")
    digit_images = [PILImage.open("./images/0_image.png"), PILImage.open("./images/1_image.png"), PILImage.open("./images/2_image.png"), PILImage.open("./images/3_image.png"),  PILImage.open("./images/4_image.png"),   PILImage.open("./images/5_image.png")]
    sand = np.asarray(img)/255	
    goal_x = 575
    goal_y = 530
    first_update = False
    done = False
    global swap
    swap = 0
    current_step = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = BoundedNumericProperty(0.0)
    #angle = float(0.0)
    #angle = ObjectProperty('float')
    rotation = BoundedNumericProperty(0.0)
    #rotation = ObjectProperty('float')
    #rotation = float(0.0)
    velocity_x = BoundedNumericProperty(0)
    velocity_y = BoundedNumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    event1 = None
    queue1_1 = None
    queue1_2 = None
    event2 = None
    queue2 = None
    event3 = None
    queue3 = None


    def move(self, rotation):
        #self.pos = Vector(*self.velocity) + self.pos
        self.x = int(self.velocity_x + self.x)
        self.y = int(self.velocity_y + self.y)
        self.pos = Vector(self.x, self.y)
        #self.pos = Vector(self.x, self.y)
        self.rotation = rotation
        self.angle = self.angle + self.rotation
 
        


# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
	
    def serve_car(self, start_event, reset_q, mode_q, state_q, action_q, next_state_reward_done_tuple_q):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        self.car.start_event = start_event
        self.car.reset_q = reset_q
        self.car.mode_q = mode_q
        self.car.state_q = state_q
        self.car.action_q = action_q
        self.car.next_state_reward_done_tuple_q = next_state_reward_done_tuple_q


    def get_state(self, img, car_img, digit_images, x, y,  car_angle, global_counter,longueur, largeur, on_road, hit_boundary, hit_goal, full_360_degree_rotation, distance_reduced, traversal_log): 
        if x - 40 <0 or y-40 < 0 or x+40 > longueur-1 or y+40 > longueur-1:
            return np.ones((80,80))
        else:
            if hit_boundary == True:
                digit_image = digit_images[0]
            elif hit_goal == True:
                digit_image = digit_images[5]
            elif full_360_degree_rotation == True:
                digit_image = digit_images[1]
            elif on_road == 0:
                digit_image = digit_images[2]
            elif on_road == 1:
                if distance_reduced == False:
                    digit_image = digit_images[3]
                else:
                    digit_image = digit_images[4]
				
            img_crop = img.crop((x -40, y-40, x+40, y +40))
            car_rotated = car_img.rotate(car_angle)
            car_size = (32,32)
            car_rotated = car_rotated.resize(car_size, PILImage.ANTIALIAS).convert("RGBA")
            img_crop.paste(car_rotated, (48, 48), car_rotated)
			
            if digit_image is not None:
                digit_size = (32,32)
                digit_image = digit_image.resize(digit_size, PILImage.ANTIALIAS).convert("RGBA")
                img_crop.paste(digit_image, (0, 48), digit_image)
				
            if global_counter % 500 == 0:
                print("map: get_state: for image id: " + str(int(global_counter / 500) + 1) + " angle: " + str(car_angle))
                traversal_log.write("map: get_state: car angle: " + str(car_angle) + "\n")
                img_crop.save("sand_images/sand_superimposed_car_" + str(int(global_counter / 500) + 1) + ".png", "PNG")
            state_value = np.asarray(img_crop)/255	
            return state_value
			
    def get_car_angle(self, car_angle):
        if car_angle > 360:
            car_angle = car_angle % 360
        elif car_angle < -360:
            car_angle = car_angle % (-360)				
        car_angle = car_angle/360
        return car_angle
		
    
	
    def select_random_on_road_location(self):
        t = np.random.randint(60, self.width-60), np.random.randint(60, self.height-60)
        while sand[t] != 0	:
            t = np.random.randint(60, self.width-60), np.random.randint(60, self.height-60)
        return t
		
    def select_demo_location(self, eval_episode_num, traversal_log):
        traversal_log.write("select_demo_location: eval_episode_num: "+ str(eval_episode_num) + "\n")
        demo_on_road_postions=[(1031,496), (766,468), (881,424)]
        index = (eval_episode_num - 1) % len(demo_on_road_postions)
        traversal_log.write("select_demo_location: eval_episode_num: "+ str(eval_episode_num) + " index: " + str(index) )
        return demo_on_road_postions[index]		
        

	
    def update(self, dt):

        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global first_update
        global done
        global current_step
        global max_steps
        global on_road
        global on_road_count
        global off_road_count
        global eval_on_road_stats_file
        global full_eval_on_road_stats_file
        global train_on_road_stats_file
        global mode
        global boundary_hit_count
        global goal_hit_count
        global train_episode_num
        global eval_episode_num
        global eval_traversal_log
        global full_eval_traversal_log
        global train_traversal_log
        global traversal_log
        global img
        global car_img
        global global_counter
        global digit_images
        global on_road_postions
        global random_location
        global episode_total_reward
        global stop_on_hitting_goal

        longueur = self.width
        largeur = self.height
        self.car.start_event.wait()
        if done == True:
            if traversal_log is not None:
                traversal_log.flush()
				
            if current_step > 0:
                if mode == "Train" :
                    train_on_road_stats_file.write(str(train_episode_num) + "," + str(on_road_count) + "," + str(off_road_count) + "," + str(boundary_hit_count) + "," + str(goal_hit_count) +  "\n") 
                    train_on_road_stats_file.flush()
                elif mode == "Eval" :
                    eval_on_road_stats_file.write( str(train_episode_num) + "," +  str(eval_episode_num) + "," + str(on_road_count) + "," + str(off_road_count) + "," + str(boundary_hit_count) + "," + str(goal_hit_count) +  "\n")
                    eval_on_road_stats_file.flush()	
                else:
                    full_eval_on_road_stats_file.write(str(eval_episode_num) + "," + str(on_road_count) + "," + str(off_road_count) + "," + str(boundary_hit_count) + "," + str(goal_hit_count) +  "\n")
                    full_eval_on_road_stats_file.flush()	
            reset = self.car.reset_q.get()

            if reset == True:
                print("first_update is set to True")
                first_update = True
				
            (mode, train_episode_num, eval_episode_num) = self.car.mode_q.get()
            print("mode: ", mode, " train_episode_num: ",  train_episode_num, " eval_episode_num:", eval_episode_num ) 
            if mode == "Train":
                max_steps = 2500
                traversal_log = train_traversal_log
            elif mode == "Eval": 
                max_steps = 500
                traversal_log = eval_traversal_log
            else:
                max_steps = 2500
                traversal_log = full_eval_traversal_log				
			
        if first_update:
            init()
            on_road_count = 0
            off_road_count = 0
            boundary_hit_count = 0
            goal_hit_count = 0
            episode_total_reward = 0.0
            #self.car.pos = (100,100)
            #if random_location==True:
            #    self.car.pos = Vector(np.random.randint(100, longueur-100), np.random.randint(100, largeur-100))
            #else:
            #    new_index = np.random.randint(len(on_road_postions))
            #    print("After reset new_index: ", new_index)
            #    traversal_log.write("After reset new_index: " + str(new_index) + "\n")
            #    self.car.pos = on_road_postions[new_index]
			
			
			
            self.car.rotation = 0.0
            self.car.angle = 0.0
			
            if mode=="Train" or  mode == "Eval":
                self.car.pos = Vector(np.random.randint(100, longueur-100), np.random.randint(100, largeur-100))                
            elif mode == "Full_Eval":
                if full_eval_demo_mode == False:
                    self.car.pos = self.select_random_on_road_location() 
                else:
                    self.car.pos = self.select_demo_location(eval_episode_num, traversal_log)
                    				
               
            print("After reset car position: ", self.car.pos)
            traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : After reset new car position: " + str(self.car.pos) + "\n")
            #state = sand[int(self.car.x)-40:int(self.car.x)+40, int(self.car.y)-40:int(self.car.y)+40]

			
			## TODO:
            #if state.shape[0] != 80 or state.shape[1] != 80:
            #    state = np.ones((80,80))
			
            xx = goal_x - self.car.x
            yy = goal_y - self.car.y
            if sand[int(self.car.x),int(self.car.y)] > 0:
                on_road = -1
                off_road_count += 1
            else :
                on_road = 1
                on_road_count += 1
            orientation = Vector(*self.car.velocity).angle((xx,yy))/360.
            state = self.get_state( img, car_img, digit_images, self.car.x, self.car.y, self.car.angle, global_counter,longueur, largeur, 0, False, False, False, False, traversal_log)			
            car_angle = self.get_car_angle(self.car.angle) 
            self.car.state_q.put((state, np.array([orientation, car_angle, 1, on_road])))
            print("map.py self.car.state_q", self.car.state_q)
 		
   
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/360.
        hit_boundary = False
        hit_goal = False
        full_360_degree_rotation = False
        distance_reduced = False
		
		
        action_array = self.car.action_q.get()
        #action = int(np.around(action_array[0]))
        rotation = action_array[0]
        rotation = 0.6 * rotation
        velocity = action_array[1]
        new_velocity = 0.4 + 1 + velocity*0.2
        print("map: Got rotation: ", rotation, " velocity: ", new_velocity)
        traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : map: Got rotation: " + str(rotation) + " velocity: " + str(new_velocity) + "\n")
        #rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
		
        if self.car.x < 40 or self.car.x > self.width - 40 or self.car.y < 40 or self.car.y > self.height - 40:
            if mode=="Train" or  mode == "Eval":
                self.car.pos = Vector(np.random.randint(100, longueur-100), np.random.randint(100, largeur-100))                
            elif mode == "Full_Eval":
                if full_eval_demo_mode == False:
                    self.car.pos = self.select_random_on_road_location() 
                else:
                    self.car.pos = self.select_demo_location(eval_episode_num, traversal_log)
            last_reward = -50
            self.car.rotation = 0.0
            self.car.angle = 0.0
            boundary_hit_count += 1
            hit_boundary = True
            print("Hit Boundary: new car position: ",self.car.pos, " rotation: ", self.car.rotation, " angle: ", self.car.angle)
            traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : Hit Boundary: new car position: " + str(self.car.pos) +  " rotation: " + str(self.car.rotation) + " angle: " + str(self.car.angle) +  "\n")
			
        

        if sand[int(self.car.x),int(self.car.y)] > 0:
            #vel = 0.4 + np.random.uniform(0, 2)
            #self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            self.car.velocity = Vector(new_velocity, 0).rotate(self.car.angle) 
            last_reward = -2
            on_road = 0
            off_road_count += 1
            print(1,  current_step + 1, int(self.car.x), int(self.car.y), goal_x, goal_y, float(distance - last_distance),  im.read_pixel(int(self.car.x),int(self.car.y)), last_reward)
            traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : 1" + " " +  str(current_step + 1) + " " +  str (int(self.car.x)) + " " +  str(int(self.car.y)) + " " + str(goal_x) + " " +  str(goal_y) + " " +  str(float(distance - last_distance)) + " " + str(im.read_pixel(int(self.car.x),int(self.car.y))) + " " + str(last_reward) + "\n")
        else: # otherwise
            #self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            self.car.velocity = Vector(new_velocity, 0).rotate(self.car.angle)
            on_road = 1
            on_road_count += 1            
            last_reward = -0.5
            
            if distance < last_distance:
                last_reward = last_reward + 5
                distance_reduced = True
                
            else:
                last_reward = last_reward + 2
                on_road = 1
                distance_reduced = False				
            print(0, current_step + 1, int(self.car.x), int(self.car.y), goal_x, goal_y, float(distance - last_distance),  im.read_pixel(int(self.car.x),int(self.car.y)), last_reward)
            traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : 0" + " " +  str(current_step + 1) + " " +  str (int(self.car.x)) + " " +  str(int(self.car.y)) + " " + str(goal_x) + " " +  str(goal_y) + " " +  str(float(distance - last_distance)) + " " + str(im.read_pixel(int(self.car.x),int(self.car.y))) + " " + str(last_reward)+ "\n")
        

        if distance < 25:
            last_reward = 100
            
            goal_hit_count += 1
            hit_goal = True

                
            if swap == 1:
                print("Hit the Goal 2: (" + str(goal_x) + ", " + str(goal_y) + ")")
                traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : Hit the Goal 2: (" + str(goal_x) + ", " + str(goal_y) + ")\n")
                #goal_x = 1420
                #goal_y = 622
                goal_x = 575
                goal_y = 530
                swap = 0
            else:
                print("Hit the Goal 1: (" + str(goal_x) + ", " + str(goal_y) + ")")
                traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) +  " : Hit the Goal 1: (" + str(goal_x) + ", " + str(goal_y) + ")\n")
                #goal_x = 212
                #goal_y = 150
                #goal_x = 975
                #goal_y = 110
                #goal_x = 610
                #goal_y = 45
                goal_x = 610
                goal_y = 45
                swap = 1
				
            if mode == "Full_Eval" and hit_goal==True and full_eval_demo_mode==True:
                #t_color = ListProperty([1,1,0,0])
                episode_total_reward += last_reward
                #popup = Popup(title='Test popup', content=Label(text="Congratulations! your car has reached the destination and earned total rewards: " + str(episode_total_reward) + " during the trip"),  size=(200, 200), auto_dismiss=False)              
                popup = Popup(title='Test popup', content=Label(text="Congratulations! your car has reached the destination and earned total rewards: " + str(episode_total_reward) + " during the trip"),  size=(200, 200), auto_dismiss=True)              
                popup.open()
                time.sleep(3)
                popup.dismiss()				
                done = True
				
            if mode == "Full_Eval" and hit_goal==True and full_eval_demo_mode==False:
                self.car.pos = self.select_random_on_road_location()
                print("After hiting goal new car position: " + str(self.car.pos))
                traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : After hitting goal new car position: " + str(self.car.pos) + "\n")


        last_distance = distance
		
        #next_state = sand[int(self.car.x)-40:int(self.car.x)+40, int(self.car.y)-40:int(self.car.y)+40]
        next_state = self.get_state(img, car_img, digit_images, self.car.x, self.car.y, self.car.angle, global_counter,longueur, largeur, on_road, hit_boundary, hit_goal, full_360_degree_rotation, distance_reduced, traversal_log)
		## TODO:
        #if next_state.shape[0] != 80 or next_state.shape[1] != 80:
        #    next_state = np.ones((80,80))
        if self.car.angle >= 360:	
            self.car.angle = self.car.angle % 360
            last_reward += -50
            print("360 degree clockwise rotation happended rewarding -50: ")
            traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : 360 degree rotation happended rewarding -50: \n")
        elif self.car.angle <= -360:	
            self.car.angle = self.car.angle % (-360)
            last_reward += -50
            print("360 degree anti-clockwise rotation happended rewarding -50: ")
            traversal_log.write("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : 360 degree anti-clockwise rotation happended rewarding -50: \n")
        reward = last_reward
        current_step += 1
        global_counter += 1
        if done== False:
            episode_total_reward += reward
        print("Train episode: " + str(train_episode_num) + " Eval episode: " + str(eval_episode_num) + " : current_step: " + str(current_step) + " reward: " + str(reward) + " episode_total_reward: " + str(episode_total_reward))
        if current_step >= max_steps:
            done = True
        distance_diff = (distance - last_distance)/4
        car_angle = self.get_car_angle(self.car.angle)
        self.car.next_state_reward_done_tuple_q.put(((next_state, np.array([orientation, car_angle, distance_diff, on_road])), reward, done, current_step))
        #print("map.py self.car.next_state_reward_done_tuple_q", self.car.next_state_reward_done_tuple_q)
		#self.car.event2.set()
		

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):
    def __init__(self, start_event, reset_q, mode_q, state_q, action_q, next_state_reward_done_tuple_q):
        super(CarApp, self).__init__()
        self.start_event = start_event
        self.reset_q = reset_q
        self.mode_q = mode_q
        self.state_q = state_q
        self.action_q = action_q
        self.next_state_reward_done_tuple_q = next_state_reward_done_tuple_q

    def build(self):
        parent = Game()
        parent.serve_car(self.start_event, self.reset_q, self.mode_q, self.state_q, self.action_q, self.next_state_reward_done_tuple_q)
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
 

    def load(self, obj):
        print("loading last saved brain...")


# Running the whole thing
if __name__ == '__main__':
    #car_app_instance = CarApp(1,2,3)
    #car_app_instance.run()
    #print("CarApp state_dim=",car_app_instance.state_dim)
    print("Hi")
