import sys
import os
import io
import json


#yourpath = '/home/aditya/CSCI544Corpus/train/'
# yourpath = '/home/aditya/CSCI544Corpus/myData/1'


class Training:
    def __init__(self):
        self.hamFileList = []
        self.spamFileList = []
        self.spamVocabulary = dict()
        self.hamVocabulary = dict()
        self.spamSet = set()
        self.hamSet = set()
        self.setUnionDict = dict()
        self.setInterDict = dict()
        self.totalVoc = dict()
        self.spamFileCount = 0
        self.hamFileCount = 0
        self.spamWordsCount = 0
        self.hamWordsCount = 0
        self.EnglishStopWords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
    def incr_total_words(self):
        self.total_words += 1
    def readFiles(self):
        for root, dirs, files in os.walk(sys.argv[1], topdown=True):
            for presentFile in files:
                filePath = os.path.join(root, presentFile)
                if("spam" in filePath):
                    self.spamFileList.append(filePath)
                elif("ham" in filePath):
                    self.hamFileList.append(filePath)
                else:
                    pass
    def readSpamList(self):
        self.spamVocabulary.clear()
        for sampleSpamFile in self.spamFileList:
            with io.open(sampleSpamFile, "r", encoding="latin1") as sFile:
           # with open(sampleSpamFile) as sFile:
                self.spamFileCount += 1
                for textLine in sFile: 
                    textLine = textLine.split(" ")
                    for word in textLine:
                        word = word.strip("\n")
                        word = word.strip("\r")
                 #       if (word not in self.EnglishStopWords) and (word.isnumeric() is False):
                        if word in self.spamSet:
                            self.spamVocabulary[word] += 1
                        else:
                            self.spamVocabulary[word] = 1
                            self.spamSet.add(word)
    def readHamList(self):
        self.hamVocabulary.clear()
        for sampleHamFile in self.hamFileList:
            with io.open(sampleHamFile, "r", encoding="latin1") as hFile:
                self.hamFileCount += 1
                for textLine in hFile: 
                    textLine = textLine.split(" ")
                    for word in textLine:
                        word = word.strip("\n")
                        word = word.strip("\r")
                  #      if (word not in self.EnglishStopWords) and (word.isnumeric() is False):
                        if word in self.hamSet:
                            self.hamVocabulary[word] += 1
                        else:
                            self.hamVocabulary[word] = 1
                            self.hamSet.add(word)
    def clearDictionaries(self):
        self.spamVocabulary.clear()
        self.hamVocabulary.clear
        ()
    def removeFiles(self):
        mFilePath1 = "spam.txt"
        mFilePath2 = "ham.txt"
        mFilePath3 = "nbmodel_10Perc.txt"
        if (os.path.exists(mFilePath1) and os.path.exists(mFilePath2) and os.path.exists(mFilePath3)):
           os.remove(mFilePath1)
           os.remove(mFilePath2)
           os.remove(mFilePath3)
        else:
            print ("One of the file did not get deleted")
    def writeDict(self):
        spamFile = 'spam.txt' 
        hamFile = 'ham.txt'
        modelFile = 'nbmodel_10Perc.txt'
        s1 = json.dumps(self.hamVocabulary, indent=4, sort_keys=True)
        # print (s1)
#         with open(hamFile, "w") as text_file:
#             text_file.write(s1)
#         s2 = json.dumps(self.spamVocabulary, indent=4, sort_keys=True)
#         with open(spamFile, "w") as text_file:
#             text_file.write(s2)
        self.totalVoc['SPAM'] = self.spamVocabulary
        self.totalVoc['HAM'] = self.hamVocabulary
        sDict = dict()
        unionSet = self.spamSet.union(self.hamSet)
        interSet = self.spamSet.intersection(self.hamSet)
        for i in unionSet:
            self.setUnionDict[i] = 1
        for i in interSet:
            self.setInterDict[i] = 1
        for key, value in self.spamVocabulary.items():
                self.spamWordsCount += int(value)
        for key, value in self.hamVocabulary.items():
                self.hamWordsCount += int(value)
        
        sDict['UNION'] = self.setUnionDict
        sDict['INTERSECT'] = self.setInterDict
        sDict['PSPAM'] = float(self.spamFileCount) / (self.spamFileCount + self.hamFileCount)
        sDict['PHAM'] = float(self.hamFileCount) / (self.spamFileCount + self.hamFileCount)
        sDict['SVOCAB_SUM_KEY'] = self.spamWordsCount
        sDict['HVOCAB_SUM_KEY'] = self.hamWordsCount
        sDict['INTERSECT_COUNT'] = len(self.setInterDict)
        sDict['SVOCAB_COUNT_PLUS_HVOCAB_COUNT'] = len(self.spamVocabulary) + len(self.hamVocabulary)
        sDict['UNION_COUNT'] = len(self.spamVocabulary) + len(self.hamVocabulary) - len(self.setInterDict)
        sDict['HVOCAB_COUNT'] = len(self.hamVocabulary)
        sDict['SVOCAB_COUNT'] = len(self.spamVocabulary)
        
        self.totalVoc['VOCAB'] = sDict
        s3 = json.dumps(self.totalVoc, indent=4, sort_keys=True)
       # print(s3)
        with open(modelFile, 'w') as tfile:
            tfile.write(s3)
    
       
    def printDetails(self):
        print(len(self.spamFileList))
        print(len(self.hamFileList))

        
if __name__ == "__main__":
    trainingObject = Training()
    trainingObject.removeFiles()
    trainingObject.readFiles()
    trainingObject.clearDictionaries()
    trainingObject.readSpamList()
    trainingObject.readHamList()
    trainingObject.writeDict()
    trainingObject.printDetails()
    
    
#main()
