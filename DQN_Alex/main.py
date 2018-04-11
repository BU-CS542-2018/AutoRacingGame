import gym
import universe  # register the universe environments
import threading
try:
    # Python2
    import Tkinter as tk
except ImportError:
    # Python3
    import tkinter as tk


quit = False
history = []
ACTIONS = [[("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)]]



def convertHistoryToKey():
    key = ""
    global history
    if "Up" in history:
        key = key + "Up"
    elif "Down" in history:
        key = key + "Down"
        
    if "Left" in history:
        key = key + "Left"
    elif "Right" in history:
        key = key + "Right"

    return key


def getActionFromKeyPress():
    keyPressed = convertHistoryToKey()

    global ACTIONS
    if keyPressed == "Up":
        return ACTIONS[0]
    elif keyPressed == "Left":
        return ACTIONS[4]
    elif keyPressed == "Right":
        return ACTIONS[5]
    elif keyPressed == "Down":
        return ACTIONS[3]
    elif keyPressed == "UpLeft":
        return ACTIONS[1]
    elif keyPressed == "UpRight":
        return ACTIONS[2]
    elif keyPressed == "DownLeft":
        return ACTIONS[6]
    elif keyPressed == "DownRight":
        return ACTIONS[7] 
    
    #Do nothing
    return ACTIONS[8]


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


def keyup(event):
    kP = str(event.keysym)
    if kP == 'Escape':
        global quit
        quit = True
        root.destroy()
    
    global history
    if kP in history :
        history.pop(history.index(kP))


def keydown(event):
    kP = str(event.keysym)
    if kP == 'Escape':
        global quit
        quit = True
        root.destroy()
    
    if kP == "Up" or kP == "Down" or kP == "Left" or kP == "Right":
        global history
        if not kP in history :
            history.append(kP)



#Start up the car game 
task = runCarGame
t = threading.Thread(target=task)
t.start()

#Start up a GUI that can conveniently get keyboard input
root = tk.Tk()
root.bind("<KeyPress>", keydown)
root.bind("<KeyRelease>", keyup)

info = ("Welcome to the Racing Game!\n\n"
        "Use arrow keys to control car\n"
        "Exit with Esc key\n"
        "\nNOTE: This window must be in focus to play\n"
        )
l = tk.Label(root, font=("Helvetica", 20), text=info)
l.pack()

root.mainloop()


#Wait for the game thread to also quit when pressing Esc
t.join();



 
