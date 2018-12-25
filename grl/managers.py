import collections
import numpy as np
import grl.utilities

class HistoryManager:

    def __init__(self, history=[], start_timestep=0, MAX_LENGTH=None):
        self.MAX_LENGTH = MAX_LENGTH
        self.h = collections.deque(history, MAX_LENGTH)
        self.t = start_timestep + len(history)
        self.part = []
        
    def getHistory(self):
        return self.h

    def getTimestep(self):
        return self.t

    def getPartialUpdate(self):
        return self.part
    
    def setHistory(self, history, start_timestep=0):
        self.h.clear()
        self.h.append(history)
        self.t = start_timestep + len(history)

    def extendHistory(self, elem, is_complete=True):
        # the type of the elem can be different for each manager
        # in a standard grl framework, it is a (a,e) tuple
        self.part.append(elem)
        if is_complete:
            elem = tuple(self.part)
            self.part.clear()
            self.h.append(elem)
            self.t += 1

class PerceptManager:

    def __init__(self, emission_matrix=[], labels=None):
        self.E = emission_matrix
        self.labels = labels

    def setEmissionMatrix(self, emission_matrix):
        self.E = emission_matrix
    
    def setLabels(self, labels):
        self.labels = labels
    
    def getLabels(self):
        return self.labels

    def getEmissionMatrix(self):
        return self.E

    def getPercept(self, state):
        e = grl.utilities.sample(self.E[state,:])
        if self.labels:
            e = self.labels[e]
        return e


class StateManager:
    
    def __init__(self, start_state=0, transition_matrix=[], labels=None):
        self.T = transition_matrix
        self.s = start_state
        self.labels = labels
        
    def getCurrentState(self):
        s = self.s
        if self.labels:
            s = self.labels[s]
        return s

    def setCurrentState(self, state):
        self.s = state

    def setLabels(self, labels):
        self.labels = labels
    
    def getLabels(self):
        return self.labels

    def setTransitionMatrix(self, transition_matrix):
        self.T = transition_matrix

    def getTransitionMatrix(self):
        return self.T

    def makeTransition(self, action):
        self.s = grl.utilities.sample(self.T[action,self.s,:])
        return self.s