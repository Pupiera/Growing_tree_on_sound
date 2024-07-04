# This script split the wavfile on the timestamp defined in the orfeo conllu file
# It create a directory in the directory of the corpus called "NameOfCorpus_clips"
# These splited wav file are the one used by the wav2tree model


from pydub import AudioSegment
import sys
import os

def readConlluFile(path):
    firstBeginTimeStamp=-1
    lastEndTimeStamp=-1
    sent_id="UNKNWOWN"
    listTimeStamp=[]
    newSentence=True
    print(path)
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line=="\n": #end of sentence
                newSentence=True
                listTimeStamp.append([sent_id,
                                      (firstBeginTimeStamp,
                                      lastEndTimeStamp)])
            elif line.startswith("#"): #ignore commented line
                if line.startswith("# sent_id"):
                    sent_id=line.split("=")[1].rstrip("\n").rstrip(" ")
                continue

            else:
                splited=line.split("\t")
                if newSentence:
                    newSentence=False
                    firstBeginTimeStamp=float(splited[10])
                lastEndTimeStamp=float(splited[11])
    return listTimeStamp

def processDir(path):
    dirName=os.path.join(path,path.split("/")[-1]+"_clips") 
    try:
        os.makedirs(dirName) #remove .orfeo
    except FileExistsError:
        print(f"overwriting {dirName}")
    for file in os.listdir(path):
        if file.endswith(".orfeo"):
            baseName=file[0:-6]
            listTimeStamp=readConlluFile(os.path.join(path,file))
            audioFile=os.path.join(path,baseName+".wav")
            try:
                OGAudio=AudioSegment.from_wav(audioFile)
            except FileNotFoundError:
                print(f"file at path : {audioFile} does not exist")
                continue
            for ts in listTimeStamp:
                sent_id=ts[0].replace(" ","")
                times=ts[1]
                beginTime=times[0]
                endTime=times[1]
                if beginTime >= endTime:
                    print(f"Time annotation invalid. Not creating file for phrase {sent_id}. BeginTime {beginTime}, endTime : {endTime}")
                    continue
                newAudio = OGAudio[beginTime*1000:endTime*1000]
                newAudio = newAudio.set_frame_rate(16000)
                newAudio.export(os.path.join(dirName,sent_id+".wav"),format="wav")
                  
if __name__ =="__main__":
    PATH = sys.argv[1]
    print(os.listdir(PATH))
    allDir=[d for d in os.listdir(PATH) if os.path.isdir(os.path.join(PATH,d))]
    print(allDir)
    for dir in allDir:
        processDir(os.path.join(PATH,dir))
