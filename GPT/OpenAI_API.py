import UI as ui
#import Google_Speech_And_Text as g
import openai
from playsound import playsound #Play audio file, playsound version 1.3.0, right-click on playsound package and install version 1.2.2 to fix error
import sounddevice
import wavio
#import pandas
import locale
import tiktoken
import json
import logging

logging.basicConfig(filename='OpenAI_API.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
locale.setlocale( locale.LC_ALL, '' )

# Set OpenAI's API Key by reading key in file OpenAI_API_Key.txt
open_API_Key = open('OpenAI_API_Key.txt', 'r')
openai.api_key = open_API_Key.read()
open_API_Key.close()

user = list()
assistant = list()

#Ask GPT a question and get a response.  Doesn't send chat history to GPT.  Doesn't save chat history in user array
#Does NOT require inputText to be formatted as a dictionary.  Just pass in a string >>> inputText = 'this is my prompt'
def OpenAI_Chat_No_History(inputText, printAllResults):
    chat_history = list()
    chat_history.append({"role": "user", "content": inputText})

    response = openai.ChatCompletion.create(
        model = ui.api_combo_value.get(),
        messages = chat_history
    )
    
    print(inputText)
    
    return response.choices[0].message.content

#Ask GPT a question and get a response, while allowing chat history to be saved for user and GPT.
#Saves chat history in user array
#Does NOT require inputText to be formatted as a dictionary.  Just pass in a string >>> inputText = 'this is my prompt'
def OpenAI_Chat(inputText, printAllResults):
    print('OpenAI_Chat: Beginning')
    chat_history = list()
    print('OpenAI_Chat: 2')
    user.append(inputText)
    print('OpenAI_Chat: 3')
    i = 0
    assistant_length = len(assistant) - 1
    user_length = len(user) - 1
    print('OpenAI_Chat: 4')
    print('assistant length: ' + str(assistant_length))
    print('user length: ' + str(user_length))

    #Calculate estimated token length
    #Determine if certain user and assistant messages need to be excluded from chat history
    #due to going above 16,000 token limit
    i = 0
    total_characters = 0
    threshold = 50000
    chat_history_threshold = 99999 #Set counter really high to include all chat history if threshold is never passed

    total_characters = len(str(user[0])) #Always need to add the first prompt that contains instructions and the outline
    i = user_length
    print('user_length: ' + str(user_length))

    while i >= 1:
        print('i: ' + str(i))
        total_characters += len(str(user[i]))

        if i <= assistant_length:
            total_characters += len(str(assistant[i]))

        i = i - 1

        if total_characters >= threshold:
            chat_history_threshold = i #Only add user and assistant chat history prior to this index number
            i = 0 #Break loop

    while i <= user_length:
        #Don't add any more user or assistant text to chat_history if it would cause going above 50,000 characters (16k tokens)
        if i == 0 or chat_history_threshold <= i or chat_history_threshold == 99999:
            print('OpenAI_Chat: i: ' + str(i))
            print('OpenAI_Chat: user_length: ' + str(user_length))
            #print(str(i) + ": " + user[i])
            print('OpenAI_Chat: user[i]: ' + str(i) + ': ' + str(user[i]))
            chat_history.append({"role": "user", "content": user[i]})
            #if(len(user) - 1 != 0): #If this is the first time ChatGPT was called and an assistant message doesn't exist yet
            #print(str(i) + ": " + assistant[i])
        
        if chat_history_threshold <= i or chat_history_threshold == 99999:    
            if i <= assistant_length:
                print('OpenAI_Chat: if i <= assistant_length: ' + str(i) + ': ' + assistant[i])
                logging.warning('OpenAI_Chat: if i <= assistant_length: ' + str(i) + ': ' + assistant[i])
                chat_history.append({"role": "assistant", "content": assistant[i]})
        
        i = i + 1

    #After the 2nd+ time calling ChatGPT, there will always be 1 more user message than assistant message
    #####if(len(user) - 1 > 0):
    #print(str(i) + ": " + user[i])
    # print('ass length: ' + str(assistant_length))
    # if assistant_length != -1: #If first loop, don't add to user list twice
    #     chat_history.append({"role": "user", "content": user[i]})
    
    print('OpenAI_Chat: 5')
    print(ui.api_combo_value.get())
    print('chat_history:')
    for item in chat_history:
        print(item)
        print('\n\n\n')


    response = openai.ChatCompletion.create(
        model = ui.api_combo_value.get(),
        #messages=[{"role": message["role"], "content": message["content"]} for message in chat_history],
        messages = chat_history #[
            #{"role": "user", "content": user[i]}
            #{"role": "assistant", "content": assistant[i]}
        #]

        # messages=[
        #     {"role": "user", "content": inputText}
        #     #In general, gpt-3.5-turbo-0301 does not pay strong attention to the system message, and therefore important instructions are often better placed in a user message.
        #     # {"role": "system", "content": "You are a helpful assistant."}, #The system message helps set the behavior of the assistant.
        #     # {"role": "user", "content": "Who won the world series in 2020?"}, #The user messages help instruct the assistant. 
        #     # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}, #The assistant messages help store prior responses. 
        #     # {"role": "user", "content": "Where was it played?"} #The user's final question.
        # ]
        #max_tokens = token_response_limit,
        #max_tokens = 1,
        #temperature = 0.7, #temperature = float(ui.temperature_textbox.get('1.0', ui.INSERT)),
        #presence_penalty = 1, #presence_penalty = float(ui.pres_penalty_textbox.get('1.0', ui.INSERT)),
        #frequency_penalty = 0 #frequency_penalty = float(ui.freq_penalty_textbox.get('1.0', ui.INSERT)),
        #best_of = int(ui.best_of_textbox.get('1.0', ui.INSERT)),
        #n = int(ui.completions_textbox.get('1.0', ui.INSERT)),
        #top_p = float(ui.top_p_textbox.get('1.0', ui.INSERT))
    )
    
    #if(printAllResults == True):
        #assistant’s reply can be extracted with response
        
        #print(['choices'][0]['message']['content'])

        #To see how many tokens are used by an API call, check the usage field in the API response (e.g., response['usage']['total_tokens']
        #print(response['usage']['total_tokens'])

        #print(response.usage.total_tokens)
        #print(response.choices[0])
        #print(response.choices[0].message)
        #print(response.choices[0].message.content)

    print('OpenAI_Chat: 6')
    assistant.append(response.choices[0].message.content)
    print('OpenAI_Chat: 7')
    print(chat_history)
    print('OpenAI_Chat: 8')

    return response.choices[0].message.content

#Ask GPT a question and get a response.  Doesn't send chat history to GPT.
#Except it DOES save chat history in user array, so function OpenAI_Chat can be called with full chat history afterwards
#Requires inputText to be formatted as a dictionary >>> inputText = [{"role": "system", "name": "example_assistant", "content": prompt_list}]
def OpenAI_Chat_Multiple_Messages(inputText):
    chat_history = list()
    user.append(inputText)

    response = openai.ChatCompletion.create(
        model = ui.api_combo_value.get(),
        messages = inputText
    )
    
    return response.choices[0].message.content

#How to count number of tokens before passing to OpenAI
#https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
#Install package tiktoken
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError('num_tokens_from_messages() is not presently implemented for model {model}.')

def test_tokens(messages, model="gpt-3.5-turbo-0301"):
    messages = [
        {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
        {"role": "system", "name":"example_user", "content": "New synergies will help drive top-line growth."},
        {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
        {"role": "system", "name":"example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
        {"role": "system", "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
        {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
    ]

    model = "gpt-3.5-turbo-0301"

    print(f"{num_tokens_from_messages(messages, model)} prompt tokens counted.")
    # Should show ~126 total_tokens

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    print(f'{response["usage"]["prompt_tokens"]} prompt tokens used.')

# Inputs user's question as a string to OpenAI's GPT-3 Completion model and returns answer as text
def OpenAI_Completion(inputText, printAllResults):
    if ui.model_combo_value.get() == 'text-davinci-003':
        cost_per_token = 0.00002
        model_max_token_input_length = 4000
        model_max_token_length = 4097
    elif ui.model_combo_value.get() == 'text-curie-001':
        cost_per_token = 0.000002
        model_max_token_input_length = 4000
        model_max_token_length = 4097
    elif ui.model_combo_value.get() == 'text-babbage-001':
        cost_per_token = 0.0000005
        model_max_token_input_length = 4000
        model_max_token_length = 4097
    elif ui.model_combo_value.get() == 'text-ada-001':
        cost_per_token = 0.0000004
        model_max_token_input_length = 2000
        model_max_token_length = 2049

    characters_per_token = 4
    buffer_tokens = 100

    inputText = inputText[:int(model_max_token_input_length * characters_per_token)] #Limit the input to 2,000 to 4,000 tokens depending on model

    token_input_estimate = int(len(inputText) / characters_per_token)
    token_response_limit = int(ui.tokens_textbox.get('1.0', ui.INSERT))
                               
    if token_response_limit + token_input_estimate > model_max_token_length:
        token_response_limit = model_max_token_length - token_input_estimate - buffer_tokens
    
    # print("begin")
    # print(token_input_estimate)
    # print(token_response_limit)
    # print("end")

    response = openai.Completion.create(
        prompt=inputText + '\n', #The question that is being asked to the AI
        model = ui.api_combo_value.get(),
        max_tokens = token_response_limit,
        temperature = float(ui.temperature_textbox.get('1.0', ui.INSERT)),
        presence_penalty = float(ui.pres_penalty_textbox.get('1.0', ui.INSERT)),
        frequency_penalty = float(ui.freq_penalty_textbox.get('1.0', ui.INSERT)),
        best_of = int(ui.best_of_textbox.get('1.0', ui.INSERT)),
        n = int(ui.completions_textbox.get('1.0', ui.INSERT)),
        top_p = float(ui.top_p_textbox.get('1.0', ui.INSERT))
        # stream
        # echo
        # suffix = ui.suffix_textbox.get('1.0', ui.INSERT),
        # stop = ui.stop_1_textbox.get('1.0', ui.INSERT),
        # stop = ui.stop_2_textbox.get('1.0', ui.INSERT),
        # stop = ui.stop_3_textbox.get('1.0', ui.INSERT),
        # stop = ui.stop_4_textbox.get('1.0', ui.INSERT),
        # logit_bias = ui.logit_bias_textbox.get('1.0', ui.INSERT),
        # logprobs = ui.log_probs_textbox.get('1.0', ui.INSERT),
        # user = ui.user_textbox.get('1.0', ui.INSERT)

        #model="text-davinci-002", #More expensive, best model, good for creative stories or very complicated questions
        # temperature=0, #Always same response/more realistic
        # temperature=1, #Very creative, different responses each time


        # model="text-ada-001", #1,000 times less expensive, good for easy questions/lists/etc.
        # temperature=0.7, #More creative, but not as much as 1
        # max_tokens=200, #Returns a max of 200 words minus the number of words in the inputText.  More words cost more money.
        # top_p=1,
        # frequency_penalty=0, #Increase closer to 2 if keep getting same response back multiple times in same API call
        # presence_penalty=0
    )

    print(response.usage.total_tokens)
    print(response.usage.total_tokens * cost_per_token)

    #Update cost and total cost labels to track how much GPT-3 costs    
    cost = response.usage.total_tokens * float(ui.best_of_textbox.get('1.0', ui.INSERT)) * int(ui.completions_textbox.get('1.0', ui.INSERT)) * cost_per_token
    ui.cost_value_label['text'] = locale.currency(cost, grouping = True)
    ui.total_cost_value_label['text'] = locale.currency(float(ui.total_cost_value_label['text'].replace(',','').replace('$','')) + cost, grouping=True)

    if printAllResults == True:
        print("--------------------------------------------------")
        print("Step 2. OpenAI Method Results: Inputs user's question as a string to OpenAI's GPT-3 Completion model and returns answer as text.\n")
        print('response: ' + str(response) + '\n')
        print('response.choices[0].text: ' + response.choices[0].text + '\n')

    return response.choices[0].text.strip()

# Convert text to .wav file.
def AskQuestion(outputFile, speaker_Name, language, gender, playSound, exportAudioResponse, response_textbox, INSERT, printAllResults):
    playsound('BeginSpeaking.wav')
    freq = 24000
    duration = 5 # Number of seconds to record user audio for
    recording = sounddevice.rec(int(duration * freq), samplerate=freq, channels=1) # Start recordering user's audio through mic input
    sounddevice.wait() # Listen for X [duration] seconds before stopping
    playsound('StopSpeaking.wav')
    wavio.write('AudioOutput.wav', recording, freq, sampwidth=2)
    QuestionInput = g.SpeechToText(printAllResults) #Returns user's question in string format using Google Cloud Speech To Text API

    print("QuestionInput: " + str(QuestionInput))

    if QuestionInput is None:
        QuestionInput = response_textbox.get("1.0", INSERT) + '\n'
    else:
        if g.promptType.get() == 1: #Continue
            if response_textbox.get("1.0", INSERT) != '':
                QuestionInput = g.response_textbox.get("1.0", INSERT) + '\n' + QuestionInput #1.0 = Line 1 Character 0
            else: #Don't insert new line
                QuestionInput = QuestionInput #1.0 = Line 1 Character 0

    # def chatgpt():
    
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": "What is Python?"}],
    #     stream=True,
    #     temperature=0)
    
    #print(promptType.get())
    #print(QuestionInput)

    AnswerOutput = OpenAI_Completion(QuestionInput, printAllResults) #Passes user's question to OpenAI API and returns response in string format
    #TextToSpeech(AnswerOutput) #Passes in OpenAI's response in a string format and then saves AudioOutput.wav using Google Cloud's Text to Speech API

    if exportAudioResponse == True:
        g.TextToSpeech(AnswerOutput, outputFile, speaker_Name, language, gender, printAllResults)

    if playSound == True:
        playsound('AudioOutput.wav') #Plays OpenAI's response in audio format

    if printAllResults == True:
        print("--------------------------------------------------")
        print("Step 4. Results of AskQuestion Method: Calls all other methods.\n")

    response_textbox.delete("1.0", "end")

    return QuestionInput + AnswerOutput

    # if includeType.get() == 1: #Insert question and answer    
    #     return QuestionInput + AnswerOutput
    # else:
    #     return AnswerOutput
    
def SaveTxtFile(fileName, textToConvert):
    # save_path = 'C:/example/'
    # name_of_file = raw_input("What is the name of the file: ")
    # completeName = os.path.join(save_path, name_of_file+".txt")
    file1 = open(fileName, "w")
    #toFile = raw_input("Write what you want into the field")
    file1.write(textToConvert.text)
    file1.close()
    
"""
#Convert audio input to audio file output
def SpeechToText2():
    speech_client = speech.SpeechClient()

    bucket_name = 'fitfunapps.com'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob('CraiyonTutorial.wav')

    #bucketFolder = environ.get('GCP_BUCKET_FOLDER_NAME')
    #files = bucket.list_blobs(prefix=bucketFolder)
    files = bucket.list_blobs()
    fileList = [file.name for file in files if '.' in file.name]
    print(fileList)
    
    # with open('CraiyonTutorial.wav', 'wb') as f:
    #     #storage_client.download_blob_to_file(blob, f)
    #     audioOutput = storage_client.download_blob_to_file
    #     audioOutput = storage_client
    
    # fileList = list_files(bucket_name)
    # rand = randint(0, len(fileList) - 1)
    # blob = bucket.blob(fileList[rand])
    fileName = blob.name.split('/')[-1]
    audioOutput = 'abc'
    #audioOutput = blob
    #audioOutput = 'Audio Input.wav'

    with open(audioOutput, 'rb') as f1:
        byte_data_mp3 = f1.read()
    
    audio_WAV = speech.RecognitionAudio(content=byte_data_mp3)

    config_mp3 = speech.RecognitionConfig(
        # sample_rate_hertz=44100,
        encoding="LINEAR16", #https://cloud.google.com/speech-to-text/docs/encoding#audio-encodings
        sample_rate_hertz=16000,
        enable_automatic_punctuation=True,
        language_code='en-US'
    )
    # https://cloud.google.com/storage/docs/creating-buckets#storage-create-bucket-console
# Sync input too long. For audio longer than 1 min use LongRunningRecognize with a 'uri' parameter.

    audioFile = speech_client.recognize(config=config_mp3, audio=audio_WAV)

    if printAllResults == True:
        print("Step 1. Results of SpeechToText Method: Convert audio input to audio file output.\n")
        print('audioFile.results: ' + str(audioFile.results) + '\n') #To see full transcript

    for result in audioFile.results:
        if printAllResults == True:
            print('result.alternatives[0].transcript: ' + str(result.alternatives[0].transcript) + '\n')
            print('result.alternatives[0].confidence: ' + str(result.alternatives[0].confidence) + '\n')

        return result.alternatives[0].transcript
"""

# en-AU-Wavenet-B
#language_code='en-US',
# language_code='en-GB',
#Markup: https://cloud.google.com/text-to-speech/docs/ssml
#Voices: https://cloud.google.com/text-to-speech/docs/voices
# name='en-US-Wavenet-C',
# name='en-US-Wavenet-J', #Good male voice
# name='en-US-Wavenet-L', #Okay female
# name='en-GB-Wavenet-F',
#English (Australia)	Neural2	en-AU	en-AU-Neural2-A	FEMALE
#English (Australia)	WaveNet	en-AU	en-AU-Wavenet-B	MALE
#English (Australia)	WaveNet	en-AU	en-AU-Wavenet-D	MAL
#English (UK)	Neural2	en-GB	en-GB-Neural2-A	FEMALE
#English (UK)	Standard	en-GB	en-GB-Standard-F	FEM
#English (UK)	WaveNet	en-GB	en-GB-Wavenet-F	FEMALE
#English (US)	WaveNet	en-US	en-US-Wavenet-F	FEMALE
#English (US)	WaveNet	en-US	en-US-Wavenet-H
#English (US)	WaveNet	en-US	en-US-Wavenet-I	MALE

# import requests

# response = requests.get('https://coderfairy.com/wp-json/wp/v2/pages/2883')
# data = response.json()
#data = json.loads(post_data)


# print("Start")
# print(data["id"]) #2875
# print(data["date"]) #2023-04-27T00:45:56
# print(data["date_gmt"]) #2023-04-27T00:45:56
# print(data["guid"]) #{'rendered': 'https://coderfairy.com/python/basic-knowledge-of-python/'}
# print(data["modified"]) #2023-04-27T00:45:56
# print(data["modified_gmt"]) #2023-04-27T00:45:56
# print(data["slug"]) #basic-knowledge-of-python
# print(data["status"]) #publish
# print(data["type"]) #page
# print(data["link"]) #https://coderfairy.com/python/basic-knowledge-of-python/
# print(data["title"]) #{'rendered': 'Basic Knowledge of Python'}
# print(data["content"]["rendered"])
# html_string = data["content"]["rendered"]
# content = data["content"]
# print(content.replace("{'rendered': '", "").replace(", 'protected': False}", "")) #{'rendered': '<div class="button-container"><a class="button" href="https://coderfairy.com/python/Converting-States-to-Zip-Codes">&lt; Previous</a><br /><a class="button" href="https://coderfairy.com/python/Develop-Experience-in-Python-Libraries-Related-to-Data-Science">Next &gt;</a></div>\n<p> </p>\n<h1> Basic Knowledge of Python </h1>\n<p>Python is a high-level programming language that is suitable for scripting, rapid application development, and web development. It is an easy-to-learn, object-oriented language that is open source and free to use. Python has been around for over 30 years and has a large community of developers that contribute to its growth and development.</p>\n<h2>Why Learn Python?</h2>\n<p>Python is a versatile language that has a wide range of applications. It can be used for developing web applications, creating automation scripts, analyzing data, and even in machine learning. Here are some reasons why learning Python is important:</p>\n<ul>\n<li> Easy to Learn: Python has a simple syntax and is easy to learn for anyone who has some programming experience.</li>\n<li> Versatile: Python can be used in many fields, including web development, data science, automation, and machine learning.</li>\n<li> Large Community: Python has a large community of developers who actively contribute to its development.</li>\n<li> Open Source: Python is open-source and free to use, making it accessible to everyone.</li>\n</ul>\n<h3>Basic Syntax</h3>\n<p>To start programming in Python, you need to have a basic understanding of its syntax. Here are some of the basic syntax rules in Python:</p>\n<ul>\n<li> Python uses indentation instead of curly brackets to delimit blocks of code.</li>\n<li> Statements in Python do not end with semicolons.</li>\n<li> Parentheses are used for grouping expressions and function arguments.</li>\n<li> Comments start with the hash character (#) and can be used to add notes or explanations to your code.</li>\n</ul>\n<p>Here is an example of a simple Python program:</p>\n<pre>\n<code>\n# This is a simple Python program\nprint("Hello, World!")\n</code>\n</pre>\n<p>The above program will output &#8220;Hello, World!&#8221; to the console when executed.</p>\n<h3>Data Types</h3>\n<p>In Python, there are several built-in data types that you can use in your programs. Here are some of the common data types in Python:</p>\n<ul>\n<li> Numbers: Python supports integer, float, and complex number types.</li>\n<li> Strings: A sequence of characters that can be enclosed in single or double quotes.</li>\n<li> Lists: An ordered collection of items that can be of different types.</li>\n<li> Tuples: A collection of ordered, immutable items.</li>\n<li> Dictionaries: A collection of unordered key-value pairs.</li>\n<li> Sets: An unordered collection of unique items.</li>\n</ul>\n<p>Here is an example of how to use some of these data types:</p>\n<pre>\n<code>\n# Example of using data types in Python\nx = 5\ny = 3.14\nz = 5 + 2j\nname = "John Doe"\nmy_list = [1, 2, "three", 4.5]\nmy_tuple = (1, "two", 3.0)\nmy_dict = {"name": "John", "age": 30}\nmy_set = set([1, 2, 3])\n</code>\n</pre>\n<p>In the above example, we have defined variables of different data types, including numbers, strings, lists, tuples, dictionaries, and sets.</p>\n<h3>Functions and Control Structures</h3>\n<p>Functions and control structures are essential in programming as they allow you to create reusable code and control the flow of your program. Here are some of the common functions and control structures in Python:</p>\n<ul>\n<li> Functions: A block of code that can be called multiple times with different arguments.</li>\n<li> Loops: Allows you to iterate over a sequence of elements using a for or while loop.</li>\n<li> Conditional Statements: Allows you to execute code based on certain conditions, using if/else statements.</li>\n<li> Error Handling: Allows you to handle errors gracefully when they occur, using try/except clauses.</li>\n</ul>\n<p>Here is an example that uses all the above functions and control structures:</p>\n<pre>\n<code>\n# Example of using functions and control structures in Python\n\n# Define a function that takes two arguments and returns their sum\ndef add_numbers(a, b):\n    return a + b\n\n# Use a for loop to iterate over a list of numbers and add them up\nnumbers = [1, 2, 3, 4, 5]\nsum = 0\nfor number in numbers:\n    sum = add_numbers(sum, number)\n\n# Use an if statement to check if the sum is greater than 10\nif sum > 10:\n    print("The sum is greater than 10.")\nelse:\n    print("The sum is less than or equal to 10.")\n\n# Use a try/except block to handle errors gracefully\ntry:\n    result = add_numbers("two", 3)\n    print(result)\nexcept TypeError:\n    print("Cannot add string and integer.")\n</code>\n</pre>\n<p>In the above example, we have defined a function that takes two arguments and returns their sum. We have then used a for loop to iterate over a list of numbers and add them up. We have also used an if statement to check if the sum is greater than 10 and a try/except block to handle errors gracefully.</p>\n<h3>Conclusion</h3>\n<p>Python is a powerful programming language that is easy to learn and has a wide range of applications. By learning Python, you can become proficient in web development, data analysis, automation, and machine learning. With a basic understanding of Python&#8217;s syntax, data types, functions, and control structures, you can start building your own Python projects and contributing to the development of this amazing language.</p>\n<div class="button-container"><a class="button" href="https://coderfairy.com/python/Converting-States-to-Zip-Codes">&lt; Previous</a><br /><a class="button" href="https://coderfairy.com/python/Develop-Experience-in-Python-Libraries-Related-to-Data-Science">Next &gt;</a></div>\n<p></p>\n<div class="button-container"><a href="https://coderfairy.com/python/Converting-States-to-Zip-Codes">Converting States to Zip Codes</a><br /><a href="https://coderfairy.com/python/Develop-Experience-in-Python-Libraries-Related-to-Data-Science">Develop Experience in Python Libraries Related to Data Science</a></div>\n', 'protected': False}
# print(data["comment_status"]) #open
# print(data["author"]) #2

#print(data)
# from bs4 import BeautifulSoup

# soup = BeautifulSoup(html_string, 'html.parser')
# text = soup.get_text()
#print(text)









# import requests
# import xmlrpc.client

# # WordPress credentials
# username = 'BrandonPodell'
# password = 'BFUp 9tzN W0AE f6hK E1kf 97lx'
# url = 'https://coderfairy.com/xmlrpc.php'

# # Elementor credentials
# #elementor_token = "your_elementor_token"
# #elementor_url = "https://api.elementor.com"

# # Create a WordPress client
# wp = xmlrpc.client.ServerProxy(url)

# # Get the ID of the page you want to convert to Elementor
# page_title = "Network with Data Experts"
# pages = wp.wp.getPosts({
#     "post_type": "page",
#     "post_status": "publish",
#     "number": 1000
# })
# page_id = None
# for page in pages:
#     if page['post_title'] == page_title:
#         page_id = page['ID']
#         break

# print("Page ID :" + str(page_id))
# # Convert the page to Elementor
# if page_id is not None:
#     # Get the page content
#     page = wp.wp.getPage(page_id)
#     content = page['post_content']

#     # Create a new Elementor page
#     headers = {
#         "Content-Type": "application/json"#,
#         #"Authorization": f"Bearer {elementor_token}"
#     }
#     data = {
#         "title": page_title,
#         "content": content
#     }
#     response = requests.post(f"{url}/v1/pages", headers=headers, json=data)
#     elementor_page_id = response.json()["id"]

#     # Update the WordPress page to use the Elementor page
#     wp.wp.editPage(page_id, {
#         "page_template": "elementor_header_footer",
#         "elementor_library_type": "page",
#         "elementor_library_page_id": elementor_page_id
#     })



# import xmlrpc.client

# username = 'BrandonPodell'
# password = 'BFUp 9tzN W0AE f6hK E1kf 97lx'
# url = "https://coderfairy.com/xmlrpc.php"

# # Create a WordPress client
# wp = xmlrpc.client.ServerProxy(url)
# token = wp.getUsersBlogs(username, password)[0]['blogid']

#In WordPress, go to Appearance > Theme File Editor > functions.php > Add the following code
#add_filter('xmlrpc_enabled', '__return_true');

# import xmlrpc.client

# username = 'BrandonPodell'
# password = 'BFUp 9tzN W0AE f6hK E1kf 97lx'
# url = "https://coderfairy.com/xmlrpc.php"

# # Create a WordPress client
# wp = xmlrpc.client.ServerProxy(url)
# token = wp.call(
#     "wp.getUsersBlogs",
#     username,
#     password
# )[0]["blogid"]