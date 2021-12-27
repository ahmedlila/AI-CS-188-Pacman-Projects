# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from math import inf
from statistics import mean
from game import Agent



class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        totalFood=successorGameState.getNumFood()

        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newPos_ghost= successorGameState.getGhostPosition(1)
        "*** YOUR CODE HERE ***"
        #Reflex agent Depends on current state and take an action based on it
        ghost_distance= util.manhattanDistance(newPos, newPos_ghost)
        #From Grid Class
        foodLoc=[]
        for i in range(newFood.width):
            for j in range(newFood.height):
                if newFood[i][j]:
                    loc=(i,j)
                    value= util.manhattanDistance(newPos, loc)
                    foodLoc.append(value)
        if len(foodLoc)>1: #[1,2,1,4,2,5,8]
            foodDist = min(foodLoc)
        elif len(foodLoc)==1:
            foodDist = foodLoc[0]
        else:
            foodDist = 1
        if ghost_distance<=8:
            return successorGameState.getScore()+ ghost_distance *3+ 0.9/(foodDist)
        return successorGameState.getScore()+ ghost_distance *0.1+ 0.9/(foodDist)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.minimax(gameState, agentIndex=0, depth=self.depth)[1]


    def minimax(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState),None)
        elif agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)
        else:
            return self.min_value(gameState, agentIndex, depth)

    def min_value(self, gameState, agentIndex, depth):
        mini, miniAction = inf, None
        depth = depth
        legalMoves = gameState.getLegalActions(agentIndex)
        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        if next_agent==0: depth -= 1
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            nextScore, _ = self.minimax(successor, next_agent, depth)
            if nextScore < mini:
                mini, miniAction = nextScore, action
        return mini, miniAction

    def max_value(self, gameState, agentIndex, depth):
        maxi, maxiAction = -inf, None
        # depth = depth - No Need
        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        #if nextAgent == 0: depth -=1 - No need
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            nextScore, _ = self.minimax(successor, nextAgent, depth)
            if nextScore > maxi: maxi, maxiAction = nextScore, action
        return maxi, maxiAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(gameState, agentIndex=0, depth=self.depth, alpha=-inf, beta=inf)[1]


    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState),None)
        elif agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.min_value(gameState, agentIndex, depth, alpha, beta)

    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        mini, miniAction = inf, None
        depth = depth
        legalMoves = gameState.getLegalActions(agentIndex)
        next_agent = (agentIndex + 1) % gameState.getNumAgents()
        if next_agent==0: depth -= 1
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            nextScore, _ = self.alphabeta(successor, next_agent, depth, alpha, beta)
            if nextScore < mini:
                mini, miniAction = nextScore, action
                beta = min(beta, mini)
            if alpha > beta: return mini, miniAction
        return mini, miniAction

    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        maxi, maxiAction = -inf, None
        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            nextScore, _ = self.alphabeta(successor, nextAgent, depth, alpha, beta)
            if nextScore > maxi:
                maxi, maxiAction = nextScore, action
                alpha=max(alpha, maxi)
            if alpha>beta: return maxi, maxiAction #Greater Only
        return maxi, maxiAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # 16m16s: https://www.youtube.com/watch?v=jaFRyzp7yWw
        return self.value(gameState, agentIndex=0, depth=self.depth)[1]

    def value(self, gameState, agentIndex, depth):  # Dispatch Function
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)
        elif agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)
        else:
            return self.exp_value(gameState, agentIndex, depth)

    def max_value(self, gameState, agentIndex, depth):
        maxi, maxiAction = -inf, None
        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            nextScore, _ = self.value(successor, nextAgent, depth)
            if nextScore > maxi: maxi, maxiAction = nextScore, action
        return maxi, maxiAction

    def exp_value(self, gameState, agentIndex, depth):
        exp, expAction = 0, None
        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        if nextAgent == 0: depth -= 1
        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorProb = 1 / len(legalMoves)  # Equiprobable
            exp += successorProb * self.value(successor, nextAgent, depth)[0]
        return exp, expAction
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    newPos = currentGameState.getPacmanPosition()   #Pacman position after moving (newPos)
    newFood = currentGameState.getFood()    #remaining food (newFood)
    newGhostStates = currentGameState.getGhostStates()  #Ghost State
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]     #no. of moves that each ghost will remain scared

    ScaredTime = min(newScaredTimes)   #Minimum no. of moves that a ghost will remain scared

    #Distance Between Pacman and ghosts
    ghost_pacman_distance = []
    for i in range(len(newGhostStates)):
        newPos_ghost = currentGameState.getGhostPosition(i+1)
        ghost_pacman_distance.append(util.manhattanDistance(newPos, newPos_ghost))
    ghostDist = min(ghost_pacman_distance)   #Distance from Pacman to nearest ghost


    #Food distance, remaining food
    foodLoc = []
    food_remain = 0
    for i in range(newFood.width):
        for j in range(newFood.height):
            if newFood[i][j]:
                food_remain += 1   #Remainig Food
                loc = (i, j)
                value = util.manhattanDistance(newPos, loc)
                foodLoc.append(value)
    #Distance to nearest position with food
    if len(foodLoc) > 1:  # [1,2,1,4,2,5,8]
        foodDist = min(foodLoc)    #Food Distance
    elif len(foodLoc) == 1:
        foodDist = foodLoc[0]
    else:
        foodDist = 1

    """
    *If Ghosts are not scared, the distance between Pacman and ghosts will be taken into consideration
    since the ghost can eat pacman (with negative weight). 
    *While if Ghosts are scared, the distance between Pacman and ghosts will not be considered.
    *Food distance will be considered with weight such that it minimizes the distance to food.
    *Remaining food will be multiplied by negative since we want it to be minimized.    
    """
    if ScaredTime == 0:
        return (80*currentGameState.getScore()) + (-20/(ghostDist+1)) + (30 / (foodDist+1)) - (10*food_remain)
    return (80*currentGameState.getScore()) + (20/(ghostDist+1)) + (30/(foodDist+1)) + (20*ScaredTime) - (10*food_remain)



better = betterEvaluationFunction
