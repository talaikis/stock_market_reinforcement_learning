from os.path import join, dirname
from random import random
from numpy import array, ones
from math import log
from gym import spaces, Env


BASE_DIR = dirname(__file__)


class MarketEnv(Env):
    PENALTY = 1 #0.999756079

    def __init__(self, target_symbols, input_symbols, start_date, end_date, scope=60, sudden_death=-1., cumulative_reward=False):
        self.startDate = start_date
        self.endDate = end_date
        self.scope = scope
        self.sudden_death = sudden_death
        self.cumulative_reward = cumulative_reward

        self.inputSymbols = []
        self.targetSymbols = []
        self.dataMap = {}

        for symbol in (target_symbols + input_symbols):
            data = self.get_data(symbol=symbol)

            if len(list(data.keys())) > scope:
                self.dataMap[symbol] = data
                if symbol in target_symbols:
                    self.targetSymbols.append(symbol)
                if symbol in input_symbols:
                    self.inputSymbols.append(symbol)

        self.actions = [
                "LONG",
                "SHORT",
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(ones(scope * (len(input_symbols) + 1)) * -1, ones(scope * (len(input_symbols) + 1)))

        self.reset()
        self._seed()
    
    def get_data(self, symbol):
        fn = join(BASE_DIR, "data", symbol + ".csv")

        data = {}
        lastClose = 0
        lastVolume = 0
        try:
            f = open(fn, "r")
            for line in f:
                if line.strip() != "":
                    dt, openPrice, high, low, close, volume = line.strip().split(",")
                    try:
                        if dt >= self.startDate and dt <= self.endDate:
                            high = float(high) if high != "" else float(close)
                            low = float(low) if low != "" else float(close)
                            close = float(close)
                            volume = int(volume)

                            if lastClose > 0 and close > 0 and lastVolume > 0:
                                close_ = (close - lastClose) / lastClose
                                high_ = (high - close) / close
                                low_ = (low - close) / close
                                volume_ = (volume - lastVolume) / lastVolume

                                data[dt] = (high_, low_, close_, volume_)

                            lastClose = close
                            lastVolume = volume
                    except Exception as e:
                        print((e, line.strip().split(",")))
            f.close()
        except Exception as e:
            print(e)

        return data

    def _step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        self.reward = 0
        if self.actions[action] == "LONG":
            if sum(self.boughts) < 0:
                for b in self.boughts:
                    self.reward += -(b + 1)
                if self.cumulative_reward:
                    self.reward = self.reward / max(1, len(self.boughts))

                if self.sudden_death * len(self.boughts) > self.reward:
                    self.done = True

                self.boughts = []

            self.boughts.append(1.0)
        elif self.actions[action] == "SHORT":
            if sum(self.boughts) > 0:
                for b in self.boughts:
                    self.reward += b - 1
                if self.cumulative_reward:
                    self.reward = self.reward / max(1, len(self.boughts))

                if self.sudden_death * len(self.boughts) > self.reward:
                    self.done = True

                self.boughts = []

            self.boughts.append(-1.0)
        else:
            pass

        vari = self.target[self.targetDates[self.currentTargetIndex]][2]
        self.cum = self.cum * (1 + vari)

        for i in range(len(self.boughts)):
            self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * (1 + vari * (-1 if sum(self.boughts) < 0 else 1))

        self.defineState()
        self.currentTargetIndex += 1
        if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[self.currentTargetIndex]:
            self.done = True

        if self.done:
            for b in self.boughts:
                self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1
            if self.cumulative_reward:
                self.reward = self.reward / max(1, len(self.boughts))

            self.boughts = []

        return self.state, self.reward, self.done, {"dt": self.targetDates[self.currentTargetIndex], "cum": self.cum, "code": self.targetCode}

    def _reset(self):
        self.targetCode = self.targetSymbols[int(random() * len(self.targetSymbols))]
        self.target = self.dataMap[self.targetCode]
        self.targetDates = sorted(self.target.keys())
        self.currentTargetIndex = self.scope
        self.boughts = []
        self.cum = 1.

        self.done = False
        self.reward = 0

        self.defineState()

        return self.state

    def _render(self, mode='human', close=False):
        if close:
            return
        return self.state

    '''
	def _close(self):
		pass

	def _configure(self):
		pass
	'''

    def _seed(self):
        return int(random() * 100)

    def defineState(self):
        tmpState = []

        budget = (sum(self.boughts) / len(self.boughts)) if len(self.boughts) > 0 else 1.
        size = log(max(1., len(self.boughts)), 100)
        position = 1. if sum(self.boughts) > 0 else 0.
        tmpState.append([[budget, size, position]])

        subject = []
        subjectVolume = []
        for i in range(self.scope):
            try:
                subject.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][2]])
                subjectVolume.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][3]])
            except Exception as e:
                print((self.targetCode, self.currentTargetIndex, i, len(self.targetDates)))
                self.done = True
        tmpState.append([[subject, subjectVolume]])

        tmpState = [array(i) for i in tmpState]
        self.state = tmpState
