import io
from hw3_corpus_tool import *
import sys

def readDevData(directoryPath):
    global actualDict
    actualDict=dict()
    for root, dirs, files in os.walk(directoryPath):
            for fileName in files:
                tagList=[]
                filePath = os.path.join(root, fileName)
                if filePath.startswith(".") or filePath.startswith("~"):
                    continue
                dictDialogUtterance = get_utterances_from_filename(filePath)
                for i in range(0, len(dictDialogUtterance)):
                    dialogUtterance = dictDialogUtterance[i]
                    act_tag = dialogUtterance[0]
                    tagList.append(act_tag)
                actualDict[fileName]=tagList
def readOutputFile(outputFile):
    fd=open(outputFile,"r")
    global count,total
    count = 0
    total=0
    outputLines = fd.read()
    dialogues = outputLines.split("\n\n")
    for singleDialogue in dialogues:
        if not singleDialogue:
            continue
        fNameAndTags = singleDialogue.split("\n")
        fName = fNameAndTags[0].split("=")[1]
        fName = fName.replace('\"','')
        predictLabels=fNameAndTags[1:]
        actualLabels=actualDict[fName]
        for predicted,actual in zip(predictLabels,actualLabels):
            if(predicted == actual):
                count+=1
            total+=1

def printDict(myDict):
    for k,v in myDict.items():
        print(str(k) + " : "+ str(v))   
          
          
if __name__ == "__main__":
    print("Test or Dev data is being read from "+sys.argv[1])
    readDevData(sys.argv[1])
    readOutputFile(sys.argv[2])      
    print("Count of Correctly Labelled " + str(count))
    print("Total tags in the test data " + str(total))
    accuracy = ((count/total)*100)
    print("Accuracy " + str(accuracy) + "%")