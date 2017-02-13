#  
#Name : Aditya Ramachandra Desai
#USC ID : 5246-4961-15
#Script : The following script activates the standard perceptron to learn and build learning model
#Dataset 100% : /home/aditya/CSCI544Corpus/train
#Dataset 10% : /home/aditya/CSCI544Corpus/tenPrec
#

from GenericPerceptron import GenericPerceptron
import sys
from datetime import datetime

if __name__ == "__main__":
    learnPerceptron = GenericPerceptron()
    #startTime = datetime.now()
    learnPerceptron.labeldFiles = list(learnPerceptron.labelData(sys.argv[1]))
    #print(" Learning Label Fields ")
    #print(datetime.now() - startTime)
    #startTime2 = datetime.now()
    learnPerceptron.processLables()
    #print(" Processing Label Fields ")
    #print(datetime.now() - startTime2)
    learnPerceptron.writeModel()
   
