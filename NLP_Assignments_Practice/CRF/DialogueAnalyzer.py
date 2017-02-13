from hw3_corpus_tool import *
import sys
import pycrfsuite
import io


class DialogueAnalyzer:
    def __init__(self):
        self.XGlobalFeatureListTrainBasic = []
        self.XGlobalFeatureListTestBasic = []
        self.YGlobalListTrainBasic=[]
        self.YGlobalListTestBasic=[]
        self.outputDictBasic=dict()
        
        self.XGlobalFeatureListTrainAdv = []
        self.XGlobalFeatureListTestAdv = []
        self.YGlobalListTrainAdv=[]
        self.YGlobalListTestAdv=[]
        self.outputDictAdv=dict()
        
        self.trainerBasic = pycrfsuite.Trainer(verbose=False)
        self.taggerBasic = pycrfsuite.Tagger()
        
        self.trainerAdv = pycrfsuite.Trainer(verbose=False)
        self.taggerAdv = pycrfsuite.Tagger()
        
        
    def analyzeUtteranceBasic(self,directoryPath,isTrain,CRFMODELFILE):
        for root, dirs, files in os.walk(directoryPath):
            lastSpeaker = None
            for fileName in files:
                predictLabelList=[]
                filePath = os.path.join(root, fileName)
                if filePath.startswith(".") or filePath.startswith("~"):
                    continue
                dictDialogUtterance = get_utterances_from_filename(filePath)
                firstUtterance = True
                for i in range(0, len(dictDialogUtterance)):
                    label=[]
                    XFeatureList = []
                    tokenList = []
                    posList = []
                    dialogUtterance = dictDialogUtterance[i]
                    act_tag = dialogUtterance[0]
                    speaker = dialogUtterance[1]
                    pos = dialogUtterance[2]
                    text = dialogUtterance[3]
                    if act_tag is None:
                        label.append('UNKNOWN_ACTTAG')
                    else:
                        label.append(act_tag)
                    if lastSpeaker is None:
                        lastSpeaker = speaker
                    if lastSpeaker != speaker:
                        XFeatureList.append('SC')   
                    lastSpeaker = speaker
                    if firstUtterance:
                        XFeatureList.append('FU')
                        firstUtterance = False                   
                    if pos is not None:
                        for posTag in pos:
                            tokenList.append("TOKEN_"+posTag[0])
                            posList.append("POS_"+posTag[1])
                    if pos is None:
                        tokenList.append("TOKEN_"+" ")
                        posList.append("POS_"+" ")
                    if tokenList:
                        for tok in tokenList:
                            XFeatureList.append(tok)
                    if posList:
                       for pos in posList:
                            XFeatureList.append(pos) 
                    if isTrain is True:
                        self.YGlobalListTrainBasic.append(label) 
                        self.XGlobalFeatureListTrainBasic.append(XFeatureList)
                    elif isTrain is False:
                        self.taggerBasic.open(CRFMODELFILE)
                        self.YGlobalListTestBasic.append(label) 
                        self.XGlobalFeatureListTestBasic.append(XFeatureList)
                        test=[XFeatureList]
                        predicted=self.taggerBasic.tag(test)
                        predictLabelList.append(predicted)
                if isTrain is False:
                    self.outputDictBasic[fileName]=predictLabelList         
    
    def trainCRFBasic(self,modelPath):
        #self.trainerBasic.select('ap',c1 = 1.0 )
        self.trainerBasic.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 150,  # stop earlier works good wiht 150
        'feature.possible_transitions': True
        })
        for xseq, yseq in zip(self.XGlobalFeatureListTrainBasic, self.YGlobalListTrainBasic):
            xseq=[xseq]
            #print(xseq,yseq)
            self.trainerBasic.append(xseq, yseq)
        self.trainerBasic.train(modelPath)
    def predictfromCRFBasic(self,devFilePath,CRFMODELFILE):
        self.analyzeUtteranceBasic(devFilePath,False,CRFMODELFILE)
           
    def writePredictionsBasic(self,fileName):
        #print(len(self.outputDictBasic))#number of files in dev
        quotes='"'
        with io.open(fileName,"w") as rFile:
            for fName,tList in self.outputDictBasic.items():
                rFile.write("FileName="+quotes+fName+quotes+"\n")
                for tag in tList:
                    rFile.write(tag[0]+"\n")
                rFile.write("\n")
    def analyzeUtteranceAdv(self,directoryPath,isTrain,CRFMODELFILE):
        for root, dirs, files in os.walk(directoryPath):
            lastSpeaker = None
            for fileName in files:
                predictLabelList=[]
                filePath = os.path.join(root, fileName)
                if filePath.startswith(".") or filePath.startswith("~"):
                    continue
                dictDialogUtterance = get_utterances_from_filename(filePath)
                firstUtterance = True
                for i in range(0, len(dictDialogUtterance)):
                    label=[]
                    XFeatureList = []
                    tokenList = []
                    posList = []
                    dialogUtterance = dictDialogUtterance[i]
                    act_tag = dialogUtterance[0]
                    speaker = dialogUtterance[1]
                    pos = dialogUtterance[2]
                    text = dialogUtterance[3]
                    if act_tag is None:
                        label.append('UNKNOWN_ACTTAG')
                    else:
                        label.append(act_tag)
                    if lastSpeaker is None:
                        lastSpeaker = speaker
                    ##ADVANCE FEATURE - IDENTFYING CONTINOUS SPEAKER##
                    if lastSpeaker==speaker:
                        XFeatureList.append('CS')
                    if lastSpeaker != speaker:
                        XFeatureList.append('SC')   
                    lastSpeaker = speaker
                    if firstUtterance:
                        XFeatureList.append('FU')
                        firstUtterance = False                   
                    if pos is not None:
                        for posTag in pos:
                            tokenList.append("TOKEN_"+posTag[0])
                            posList.append("POS_"+posTag[1])
                    ##ADVANCE FEATURE - CONSIDERING BIAGRAM FOR TOKEN and POS##
                    ##THIS REDUCED THE ACCURACY AND HENCE COMMENTED##
#                     new2gramTokenList = []
#                     new2gramPosList = []
#                     for j in range(0, len(tokenList)-1):
#                         new2gramTokenList.append("TOKEN_" + tokenList[j])
#                         new2gramTokenList.append("TOKEN_" + tokenList[j] + tokenList[j+1])
#                     tokenList = new2gramTokenList
#                     for j in range(0, len(posList)-1):
#                         new2gramPosList.append("POS_" + posList[j])
#                         new2gramPosList.append("POS_" + posList[j] + posList[j+1])
#                     posList = new2gramPosList
                    ##ADVANCE FEATURE - ADDING FIRST AND LAST TOKEN and POS RESPECTIVELY##
                    if tokenList:
                        for tok in tokenList:
                            XFeatureList.append(tok)
                        newTokenList=[]
                        newTokenList.insert(0, 'FTOK' + tokenList[0])
                        newTokenList.append('LTOK' + tokenList[-1])
                        newTokenString = ''.join(newTokenList)
                        XFeatureList.append(newTokenString)
                    if posList:
                        for pos in posList:
                            XFeatureList.append(pos) 
                        newPOSList=[]
                        newPOSList.insert(0, 'FPOS' + posList[0])
                        newPOSList.append('LPOS' + posList[-1])
                        newPOSString = ''.join(newTokenString)
                        XFeatureList.append(newTokenString)
                    ##ADVANCE FEATURE - IDENTFYING LAST OCCURENCE##
                    if i == len(dictDialogUtterance)-1:
                        XFeatureList.append('LU')  
                    ##ADVANCE FEATURE - IDENTFYING NUMEBRS IN TEXT ##
                    if(hasNumbers(text)):
                        XFeatureList.append('NUM')
                    if isTrain is True:
                        self.YGlobalListTrainAdv.append(label) 
                        self.XGlobalFeatureListTrainAdv.append(XFeatureList)
                    elif isTrain is False:
                        self.taggerAdv.open(CRFMODELFILE)
                        self.YGlobalListTestAdv.append(label) 
                        self.XGlobalFeatureListTestAdv.append(XFeatureList)
                        test=[XFeatureList]
                        predicted=self.taggerAdv.tag(test)
                        predictLabelList.append(predicted)
                if isTrain is False:
                    self.outputDictAdv[fileName]=predictLabelList 
   
    def trainCRFAdv(self,modelPath):   
        self.trainerAdv.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 150,  # stop earlier
        'feature.possible_transitions': True
        })
        for xseq, yseq in zip(self.XGlobalFeatureListTrainAdv, self.YGlobalListTrainAdv):
            xseq=[xseq]
            #print(xseq,yseq)
            self.trainerAdv.append(xseq, yseq)
        self.trainerAdv.train(modelPath)
    def predictfromCRFAdv(self,devFilePath,CRFMODELFILE):
        self.analyzeUtteranceAdv(devFilePath,False,CRFMODELFILE)
    def writePredictionsAdv(self,fileName):
        quotes='"'
        with io.open(fileName,"w") as rFile:
            for fName,tList in self.outputDictAdv.items():
                rFile.write("FileName="+quotes+fName+quotes+"\n")
                for tag in tList:
                    rFile.write(tag[0]+"\n")
                rFile.write("\n")
def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)