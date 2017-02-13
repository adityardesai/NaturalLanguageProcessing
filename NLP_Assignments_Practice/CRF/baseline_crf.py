from DialogueAnalyzer import *
import sys

#train : /home/aditya/CSCI544Corpus/discourse/train
#dev : /home/aditya/CSCI544Corpus/discourse/dev
#Output : 



if __name__ == "__main__":
    CRFMODELFILE='swbdBasic.crfsuite'
    dialogueAnalyzerObj = DialogueAnalyzer()
    dialogueAnalyzerObj.analyzeUtteranceBasic(sys.argv[1],True,CRFMODELFILE) #INPUTDIR
    dialogueAnalyzerObj.trainCRFBasic(CRFMODELFILE)
    dialogueAnalyzerObj.predictfromCRFBasic(sys.argv[2],CRFMODELFILE) #TESTDIR
    dialogueAnalyzerObj.writePredictionsBasic(sys.argv[3]) #OUTFILE