import gym
import universe  # register the universe environments
import threading
try:
    # Python2
    import Tkinter as tk
except ImportError:
    # Python3
    import tkinter as tk


quit
keyPressed = "Up"
ACTIONS = [[("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)]]


def getActionFromKeyPress():
    global keyPressed
    global ACTIONS
    if keyPressed == "Up":
        return ACTIONS[0]
    elif keyPressed == "Left":
        #return ACTIONS[4]      #Left
        return ACTIONS[1]       #Up and left
    elif keyPressed == "Right":
        #return ACTIONS[5]      #Right
        return ACTIONS[2]       #Up and right
    elif keyPressed == "Down":
        return ACTIONS[3]
    
    return ACTIONS[0]

def runCarGame():
    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(remotes=1)  # automatically creates a local docker container

    observation_n = env.reset()
    while True:
        global quit
        if quit:
            break
        a = getActionFromKeyPress()
        action_n = [a for ob in observation_n]  # your agent here
        observation_n, reward_n, done_n, info = env.step(action_n)
        env.render()


def key(event):
    """shows key or tk code for the key"""
    if event.keysym == 'Escape':
        global quit
        quit = True
        root.destroy()
    
    global keyPressed
    keyPressed = str(event.keysym)
    #print(keyPressed)
        

quit = False

#Start up the car game 
task = runCarGame
t = threading.Thread(target=task)
t.start()


#Start up a GUI that can conveniently get keyboard input
root = tk.Tk()
root.bind_all('<Key>', key)

info = ("Welcome to the Racing Game!\n\n"
        "Controls:\n"
        "Use arrow keys (don't have to hold down the key)\n"
        "Exit with Esc key\n"
        "\nNOTE: This screen must be in focus to play\n"
        )
l = tk.Label(root, font=("Helvetica", 20), text=info)
l.pack()

root.mainloop()

t.join();



 
