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
        newGhostStates = successorGameState.getGhostStates()
        newCapsule = currentGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        """Calculating distance to the farthest food pellet"""
        newFoodList = newFood.asList()
        min_food_distance = -1
        for food in newFoodList:
            distance = util.manhattanDistance(newPos, food)
            if min_food_distance >= distance or min_food_distance == -1:
                min_food_distance = distance

        min_capsule_distance = -1
        for capsule in newCapsule:
            distance = util.manhattanDistance(newPos, capsule)
            if min_capsule_distance >= distance or min_capsule_distance == -1:
                min_capsule_distance = distance
        if min_capsule_distance == 0: min_capsule_distance = 0.1

        """Calculating the distances from pacman to the ghosts. Also, checking for the proximity of the ghosts (at distance of 1) around pacman."""
        distances_to_ghosts = 1
        proximity_to_ghosts = 0
        for ghost_state in newGhostStates:
            distance = util.manhattanDistance(newPos, ghost_state.getPosition())
            if ghost_state.scaredTimer == 0:
                distances_to_ghosts += distance
                if distance <= 1:
                    proximity_to_ghosts += 1
            else:
                distances_to_ghosts -= distance
                proximity_to_ghosts -= 4 / ghost_state.scaredTimer

        if distances_to_ghosts == 0: distances_to_ghosts = -1

        """Combination of the above calculated metrics."""
        return successorGameState.getScore() + (4 / float(min_food_distance)) - (3 / float(distances_to_ghosts)) - proximity_to_ghosts + (5 / float(min_capsule_distance))

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
        "*** YOUR CODE HERE ***"
        return self.min_max(gameState, 0, 0, "max")[0]

    def minimax(self, gameState, agentIndex, depth):
        if depth is self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if agentIndex is 0:
            return self.min_max(gameState, agentIndex, depth, "max")[1]
        else:
            return self.min_max(gameState, agentIndex, depth, "min")[1]

    def min_max(self, gameState, agentIndex, depth, type):
        if type == "max":
            bestAction = (type, float("-inf"))
        elif type == "min":
            bestAction = (type, float("inf"))

        for action in gameState.getLegalActions(agentIndex):
            succAction = (action, self.minimax(gameState.generateSuccessor(agentIndex,action), (depth + 1)%gameState.getNumAgents(), depth+1))
            bestAction = eval(type)(bestAction,succAction,key=lambda x:x[1])

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.min_max(gameState, 0, 0, float("-inf"), float("inf"), "max")[0]

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth is self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if agentIndex is 0:
            return self.min_max(gameState, agentIndex, depth, alpha, beta, "max")[1]
        else:
            return self.min_max(gameState, agentIndex, depth, alpha, beta, "min")[1]

    def min_max(self, gameState, agentIndex, depth, alpha, beta, type):
        if type == "max":
            bestAction = (type, float("-inf"))
        elif type == "min":
            bestAction = (type, float("inf"))

        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.alphabeta(gameState.generateSuccessor(agentIndex,action), (depth + 1)%gameState.getNumAgents(), depth+1, alpha, beta))
            bestAction = eval(type)(bestAction,succAction,key=lambda x:x[1])

            # Prunning
            if type == "max":
                if bestAction[1] > beta: return bestAction
                else: alpha = eval(type)(alpha,bestAction[1])
            elif type =="min":
                if bestAction[1] < alpha: return bestAction
                else: beta = eval(type)(beta, bestAction[1])

        return bestAction

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
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, "expect", maxDepth, 0)[0]

    def expectimax(self, gameState, action, depth, agentIndex):

        if depth is 0 or gameState.isLose() or gameState.isWin():
            return (action, self.evaluationFunction(gameState))

        # if pacman (max agent) - return max successor value
        if agentIndex is 0:
            return self.maxvalue(gameState,action,depth,agentIndex)
        # if ghost (EXP agent) - return probability value
        else:
            return self.expvalue(gameState,action,depth,agentIndex)

    def maxvalue(self,gameState,action,depth,agentIndex):
        bestAction = ("max", float('-inf'))

        for legalAction in gameState.getLegalActions(agentIndex):
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            succAction = None

            if depth != self.depth * gameState.getNumAgents():
                succAction = action
            else:
                succAction = legalAction

            succValue = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction), succAction,depth - 1,nextAgent)
            bestAction = max(bestAction,succValue,key = lambda x:x[1])

        return bestAction

    def expvalue(self,gameState,action,depth,agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)
        averageScore = 0
        propability = 1.0/len(legalActions)

        for legalAction in legalActions:
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            bestAction = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction), action, depth - 1, nextAgent)
            averageScore += bestAction[1] * propability

        return (action, averageScore)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    """Calculating distance to the closest food pellet"""
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    newCapsule = currentGameState.getCapsules()

    min_food_distance = -1
    for food in newFoodList:
        distance = util.manhattanDistance(newPos, food)
        if min_food_distance >= distance or min_food_distance == -1:
            min_food_distance = distance

    """Calculating the distances from pacman to the ghosts. Also, checking for the proximity of the ghosts (at distance of 1) around pacman."""
    distances_to_ghosts = 1
    proximity_to_ghosts = 0
    for ghost_state in newGhostStates:
        distance = util.manhattanDistance(newPos, ghost_state.getPosition())
        if ghost_state.scaredTimer == 0:
            distances_to_ghosts += distance
            if distance <= 1:
                proximity_to_ghosts += 1
        else:
            distances_to_ghosts = -0.1
            proximity_to_ghosts = -ghost_state.scaredTimer

    if distances_to_ghosts == 0: distances_to_ghosts = -1

    """Obtaining the number of capsules available"""
    max_capsule_distance = float("inf")
    for capsule in newCapsule:
        distance = util.manhattanDistance(newPos, capsule)
        if max_capsule_distance < distance or max_capsule_distance == float("inf"):
            max_capsule_distance = distance

    if max_capsule_distance == 0: max_capsule_distance = 0.1

    """Combination of the above calculated metrics."""
    return currentGameState.getScore() + (7 / float(min_food_distance)) - (8 / float(distances_to_ghosts)) - proximity_to_ghosts + (20 / float(max_capsule_distance))

# Abbreviation
better = betterEvaluationFunction
