from DialogueAnalyzer import *
import sys

#train : /home/aditya/CSCI544Corpus/discourse/train
#dev : /home/aditya/CSCI544Corpus/discourse/dev
#Output : 



if __name__ == "__main__":
    CRFMODELFILE='swbdAdv.crfsuite'
    dialogueAnalyzerObj = DialogueAnalyzer()
    dialogueAnalyzerObj.analyzeUtteranceAdv(sys.argv[1],True,CRFMODELFILE) #INPUTDIR
    dialogueAnalyzerObj.trainCRFAdv(CRFMODELFILE)
    dialogueAnalyzerObj.predictfromCRFAdv(sys.argv[2],CRFMODELFILE) #TESTDIR
    dialogueAnalyzerObj.writePredictionsAdv(sys.argv[3]) #OUTFILE