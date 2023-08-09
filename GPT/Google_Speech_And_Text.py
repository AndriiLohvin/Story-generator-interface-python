import OpenAI_API as o
from google.cloud import texttospeech #Text to Speech, https://cloud.google.com/text-to-speech/docs/voices
from google.cloud import speech #Speech to Text
import os
import os.path

# Set Google Cloud's API Key for SpeechToText and TextToSpeech API's
# Reads key in file named Google_Cloud_API_Key.json
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Google_Cloud_API_Key.json'

#Convert audio input to audio file output
def SpeechToText(printAllResults):
    speech_client = speech.SpeechClient()
    audioOutput = 'AudioOutput.wav'

    with open(audioOutput, 'rb') as f1:
        byte_data_mp3 = f1.read()
    
    audio_WAV = speech.RecognitionAudio(content=byte_data_mp3)

    config_mp3 = speech.RecognitionConfig(
        sample_rate_hertz=24000,
        encoding="LINEAR16",
        enable_automatic_punctuation=True,
        language_code='en-US'
    )

    audioFile = speech_client.recognize(config=config_mp3, audio=audio_WAV)

    if printAllResults == True:
        print("Step 1. Results of SpeechToText Method: Convert audio input to audio file output.\n")
        print('audioFile.results: ' + str(audioFile.results) + '\n') #To see full transcript

    for result in audioFile.results:
        if printAllResults == True:
            print('result.alternatives[0].transcript: ' + str(result.alternatives[0].transcript) + '\n')
            print('result.alternatives[0].confidence: ' + str(result.alternatives[0].confidence) + '\n')

        return result.alternatives[0].transcript

#Convert text to .wav file
def TextToSpeech(inputText, outputFile, speaker_Name, language, gender, printAllResults):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=inputText)

    if gender == 'MALE':
        voiceGender = texttospeech.SsmlVoiceGender.MALE
    else:
        voiceGender = texttospeech.SsmlVoiceGender.FEMALE

    voice = texttospeech.VoiceSelectionParams(
        # language_code='en-GB',
        # name='en-GB-Wavenet-F',
        language_code = language,
        name = speaker_Name,
        ssml_gender = voiceGender)

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    with open(outputFile, 'wb') as out:
        out.write(response.audio_content)

    if printAllResults == True:
        print("--------------------------------------------------")
        print("Step 3. TextToSpeech Method Finished Running: Convert text to .wav file.\n")

def transcribe_gcs(gcs_uri, fileName):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        #encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=9999)
    # SaveTxtFile(fileName, response.results)
    # print(response.results)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    with open(fileName, 'a') as file:
        for result in response.results:
            file.write(result.alternatives[0].transcript + '.\n')
            #file.write('\n')
        # The first alternative is the most likely one for this portion.
        # print(u"Transcript: {}".format(result.alternatives[0].transcript))
        # print("Confidence: {}".format(result.alternatives[0].confidence))