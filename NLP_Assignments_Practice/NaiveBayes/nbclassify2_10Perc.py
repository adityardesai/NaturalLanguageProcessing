import os
import io
import json
import math
import sys

# testPath = "/home/aditya/CSCI544Corpus/myData/myDevData"
#testPath = "/home/aditya/CSCI544Corpus/dev"

class NaiveClassification:
    def __init__(self):
        self.modelVoc = dict()
        self.modelSpamVoc = dict()
        self.modelHamVoc = dict()
        
        self.modelSpamVoCount = 0
        self.modelHamVoCount = 0
        self.modelUniqueCount = 0
        self.modelPSPAM = 0
        self.modelPHAM = 0
        
        self.devFileList = []
        self.actualSpamInDev = 0
        self.actualHamInDev = 0
        self.spamResult = []
        self.hamResult = []
        
        
    def readModel(self):
        with open("nbmodel_10Perc.txt") as data_file:    
            data = json.load(data_file)
        self.modelVoc = data['VOCAB']
        self.modelHamVoc = data['HAM']
        self.modelSpamVoc = data['SPAM']
        self.modelSpamVoCount = self.modelVoc.get('SVOCAB_SUM_KEY')
        self.modelHamVoCount = self.modelVoc.get('HVOCAB_SUM_KEY')
        self.modelUniqueCount = self.modelVoc.get('UNION_COUNT')
        self.modelPSPAM = self.modelVoc.get('PSPAM')
        self.modelPHAM = self.modelVoc.get('PHAM')
        
    def printModel(self):
        print(self.modelVoc)
        print(self.modelHamVoc)
        print(self.modelSpamVoc)
        print(self.modelSpamVoCount)
        print(self.modelHamVoCount)
        print(self.modelUniqueCount)
    def calcSpamProbability(self, spamProbability, fileContent):
        tokens = fileContent.split()
        spamWordCounts = [self.modelSpamVoc[token] if token in self.modelSpamVoc else 0 if token in self.modelHamVoc else -1 for token in tokens]
        spamWordCounts = [wordCounts for wordCounts in spamWordCounts if wordCounts != -1]
        spamProbab = [math.log((wordCounts + 1) / (self.modelSpamVoCount + self.modelUniqueCount)) for wordCounts in spamWordCounts]
        totalSpamProbab = sum(spamProbab) + spamProbability
       # print("getSpamProb " + str(totalSpamProbab))
        return totalSpamProbab

    def calcSpamProbability(self, spamProbability, fileContent):
        tokens = fileContent.split()
        spamProbab = 0
        for token in tokens:
            if token in self.modelSpamVoc:
                spamWordCounts = self.modelSpamVoc[token] 
            elif token in self.modelHamVoc:
                 spamWordCounts = 0
            else:
                # pass
                spamWordCounts = 0
            ans = math.log((spamWordCounts + 1) / (self.modelSpamVoCount + self.modelUniqueCount))
            spamProbab = spamProbab + ans
        totalSpamProbab = spamProbab + spamProbability
       # print("getSpamProb " + str(totalSpamProbab))
        return totalSpamProbab
    def calcHamProbability(self, hamProbability, fileContent):
        tokens = fileContent.split()
        hamProbab = 0
        for token in tokens:
            if token in self.modelHamVoc:
                hamWordCounts = self.modelHamVoc[token] 
            elif token in self.modelSpamVoc:
                 hamWordCounts = 0
            else:
                # pass
                hamWordCounts = 0
            ans = math.log((hamWordCounts + 1) / (self.modelHamVoCount + self.modelUniqueCount))
            hamProbab = hamProbab + ans
        totalHamProbab = hamProbab + hamProbability
       # print("getSpamProb " + str(totalSpamProbab))
        return totalHamProbab
      
    def readDevData(self):
        hCount = 0
        sCount = 0
        with io.open("nboutput_10perc.txt", "w", encoding='utf-8', errors='ignore') as rFile:
            for sampleDevFile in self.devFileList:
                with io.open(sampleDevFile, "r", encoding='latin1') as sFile:
                    content = sFile.read()
                    fileSpamProb = self.calcSpamProbability(math.log(self.modelPSPAM), content)
                    fileHamProb = self.calcHamProbability(math.log(self.modelPHAM), content)
                    if fileHamProb > fileSpamProb:
                        rFile.write("ham " + sampleDevFile + "\n")
                        self.hamResult.append(sampleDevFile)
                        hCount += 1
          #              print ("HAM")
                    elif fileHamProb < fileSpamProb:
                        rFile.write("spam " + sampleDevFile + "\n")  
                        self.spamResult.append(sampleDevFile)
                        sCount += 1
         #               print("SPAM")
                    else:
                        pass  
	#
    # The following method is used to calculate the Precision, Recall and F1 Score
    # The formulas and the calculations are take from Class slides and also from 
    # Stanford NLP lectures - https://web.stanford.edu/class/cs124/lec/naivebayes.pdf esp the truth table or 	  
    # confusion matrix                      
    def printDetails(self):
            predictResults = []
            accuracy = 0.0
            precisionHam = 0.0
            precisionSpam = 0.0
            recallHam = 0.0
            recallSpam = 0.0
            hamInHam = 0.0
            spamInHam = 0.0
            spamInSpam = 0.0 
            hamInSpam = 0.0
            FScoreHam = 0.0
            FScoreSpam = 0.0
         
            with io.open("nboutput_10perc.txt", "r", encoding='utf-8') as rFile:
                for line in rFile:
                    whiteSpaceIndex = line.find(" ")
                    predictLabel = line[0:whiteSpaceIndex]
                    filePath = line[whiteSpaceIndex + 1:]
                    fileName = filePath[filePath.rfind("/") + 1:]
            # all test files contain either "ham" or "spam" in their name
                    if "ham" in fileName:
                        actualLabel = "ham"
                    else:
                        actualLabel = "spam"
                    predictResults = predictResults + [(predictLabel, actualLabel)]
            truthTable = [[0, 0], [0, 0]]
            for result in predictResults:
                if(result[0] == result[1] == "ham"):
                    truthTable[0][0] = truthTable[0][0] + 1
                elif(result[0] == result[1] == "spam"):
                    truthTable[1][1] = truthTable[1][1] + 1
                elif(result[0] == "ham" and result[1] == "spam"):
                    truthTable[1][0] = truthTable[1][0] + 1
                else:
                    truthTable[0][1] = truthTable[0][1] + 1

            print(truthTable)
            precisionHam = truthTable[0][0] / (truthTable[0][0] + truthTable[1][0])
            recallHam = truthTable[0][0] / (truthTable[0][0] + truthTable[0][1])
            fscoreHam = 2 * precisionHam * recallHam / (precisionHam + recallHam)
            print("[HAM]Precision:{0},Recall:{1},FScore:{2}".format(precisionHam,recallHam,fscoreHam))
            precisionSpam = truthTable[1][1] / (truthTable[1][1] + truthTable[0][1])
            recallSpam = truthTable[1][1] / (truthTable[1][1] + truthTable[1][0])
            fscoreSpam = 2 * precisionSpam * recallSpam / (precisionSpam + recallSpam)
            print("[SPAM]Precision:{0},Recall:{1},FScore:{2}".format(precisionSpam,recallSpam,fscoreSpam))
if __name__ == "__main__":
    classificationObject = NaiveClassification()
    classificationObject.readModel()
    # classificationObject.printModel()
    classificationObject.populateDevList()
    classificationObject.readDevData()
    #classificationObject.printDetails()
  

#main()
