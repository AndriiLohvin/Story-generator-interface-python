import OpenAI_API as o
#import Create_WordPress_Page as p
from tkinter import *
import tkinter.ttk as tkrttk
import tkinter as tkr
import tkinter.font as TkFont
from tkinter import simpledialog
from datetime import date
import os
import openai
import logging
import locale
# from OpenAI_API import OpenAI_Chat_No_History

#Settings
padding = 20
padding2 = 10
padding3 = 5
bgcolor = '#F2F2F2'
fgcolor = '#002060'
printAllResults = False
label_font_type = ("Arial", 14)
label_font_type2 = ("Arial", 11)
textbox_font = ("Arial", 14)
#textbox_font = ("Comic Sans MS", 14, "bold")
width_of_labels = 120
height_of_labels = 30
width_of_listbox = 500 - padding
button_width = 100
button_height = 30
inputText = []

#openAI_API functions

openai.api_key = "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"

user = list()
assistant = list()

def OpenAI_Chat_No_History(inputText, printAllResults):
    chat_history = list()
    chat_history.append({"role": "user", "content": inputText})

    response = openai.ChatCompletion.create(
        model = api_combo_value.get(),
        messages = chat_history
    )
    
    print(inputText)
    
    return response.choices[0].message.content


def OpenAI_Completion(inputText, printAllResults):
    if model_combo_value.get() == 'text-davinci-003':
        cost_per_token = 0.00002
        model_max_token_input_length = 4000
        model_max_token_length = 4097
    elif model_combo_value.get() == 'text-curie-001':
        cost_per_token = 0.000002
        model_max_token_input_length = 4000
        model_max_token_length = 4097
    elif model_combo_value.get() == 'text-babbage-001':
        cost_per_token = 0.0000005
        model_max_token_input_length = 4000
        model_max_token_length = 4097
    elif model_combo_value.get() == 'text-ada-001':
        cost_per_token = 0.0000004
        model_max_token_input_length = 2000
        model_max_token_length = 2049

    characters_per_token = 4
    buffer_tokens = 100

    inputText = inputText[:int(model_max_token_input_length * characters_per_token)] #Limit the input to 2,000 to 4,000 tokens depending on model

    token_input_estimate = int(len(inputText) / characters_per_token)
    token_response_limit = int(tokens_textbox.get('1.0', INSERT))
                               
    if token_response_limit + token_input_estimate > model_max_token_length:
        token_response_limit = model_max_token_length - token_input_estimate - buffer_tokens
    
    # print("begin")
    # print(token_input_estimate)
    # print(token_response_limit)
    # print("end")

    response = openai.Completion.create(
        prompt=inputText + '\n', #The question that is being asked to the AI
        model = api_combo_value.get(),
        max_tokens = token_response_limit,
        temperature = float(temperature_textbox.get('1.0', INSERT)),
        presence_penalty = float(pres_penalty_textbox.get('1.0', INSERT)),
        frequency_penalty = float(freq_penalty_textbox.get('1.0', INSERT)),
        best_of = int(best_of_textbox.get('1.0', INSERT)),
        n = int(completions_textbox.get('1.0', INSERT)),
        top_p = float(top_p_textbox.get('1.0', INSERT))
        # stream
        # echo
        # suffix = suffix_textbox.get('1.0', INSERT),
        # stop = stop_1_textbox.get('1.0', INSERT),
        # stop = stop_2_textbox.get('1.0', INSERT),
        # stop = stop_3_textbox.get('1.0', INSERT),
        # stop = stop_4_textbox.get('1.0', INSERT),
        # logit_bias = logit_bias_textbox.get('1.0', INSERT),
        # logprobs = log_probs_textbox.get('1.0', INSERT),
        # user = user_textbox.get('1.0', INSERT)

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
    cost = response.usage.total_tokens * float(best_of_textbox.get('1.0', INSERT)) * int(completions_textbox.get('1.0', INSERT)) * cost_per_token
    cost_value_label['text'] = locale.currency(cost, grouping = True)
    total_cost_value_label['text'] = locale.currency(float(total_cost_value_label['text'].replace(',','').replace('$','')) + cost, grouping=True)

    if printAllResults == True:
        print("--------------------------------------------------")
        print("Step 2. OpenAI Method Results: Inputs user's question as a string to OpenAI's GPT-3 Completion model and returns answer as text.\n")
        print('response: ' + str(response) + '\n')
        print('response.choices[0].text: ' + response.choices[0].text + '\n')

    return response.choices[0].text.strip()


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
    print(api_combo_value.get())
    print('chat_history:')
    for item in chat_history:
        print(item)
        print('\n\n\n')


    response = openai.ChatCompletion.create(
        model = api_combo_value.get(),
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
        #temperature = 0.7, #temperature = float(temperature_textbox.get('1.0', INSERT)),
        #presence_penalty = 1, #presence_penalty = float(pres_penalty_textbox.get('1.0', INSERT)),
        #frequency_penalty = 0 #frequency_penalty = float(freq_penalty_textbox.get('1.0', INSERT)),
        #best_of = int(best_of_textbox.get('1.0', INSERT)),
        #n = int(completions_textbox.get('1.0', INSERT)),
        #top_p = float(top_p_textbox.get('1.0', INSERT))
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
        model = api_combo_value.get(),
        messages = inputText
    )
    
    return response.choices[0].message.content


#Change api_combo values when type_combo changes
def click_type_combo(event):
    if type_combo.get() == 'Text':
        api_combo['values'] = ['gpt-4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo', 'GPT-3']
    elif type_combo.get() == 'Image':
        api_combo['values'] = ['Dall-E 2', 'Stable Diffusion']
    elif type_combo.get() == 'Video':
        api_combo['values'] = ['Synthesia']
    elif type_combo.get() == 'Audio':
        api_combo['values'] = ['Google']
	
    api_combo.current(0)

#Save API Key Button: Allows user to type API key into input box and then saves it to txt file in location
#of application with file name based on value in API drop-down (2nd drop-down at top). 
def Click_save_api_key_button(event):
    print("Click_save_api_key_button")

def Click_Generate_Prompt_List():
    if prompt_textbox.get('1.0', INSERT) != '':
        prompt_listbox.delete(0, END)
        
        if api_combo_value.get() == 'gpt-3.5-turbo':
            prompt_list = OpenAI_Chat_No_History(prompt_textbox.get('1.0', INSERT), True)
            print('prompt_list: ' + prompt_list)
        else:
            prompt_list = OpenAI_Completion(prompt_textbox.get('1.0', INSERT), True)

    #list_items = [11,22,33,44, 55, 66, 77, 88, 999, 10194, 1832894, 130823, 12084]
    #txt = "hello, I am 26 years old"
    
    list_items = prompt_list.split("|")
    
    for item in list_items:
        prompt_listbox.insert('end', item)
    
    # Remove duplicates and update the listbox
    unique_items = list(set(list_items))
    list_items = []
    for item in unique_items:
        list_items.insert(END, item)
    
    #prompt_listbox.insert(1, 'first')
    #prompt_listbox.insert(2, 'second')
    #prompt_listbox.delete(2)

#Submit: Sends prompt to GPT-3 and returns text.
def Click_submit_button():
    print('click submit button')
    print('click submit button')
    print('click submit button')
    if prompt_textbox.get('1.0', INSERT) != '':
        print('before delete')
        response_textbox.delete('1.0', INSERT)
        print('after delete')

        if api_combo_value.get() == 'gpt-3.5-turbo' or api_combo_value.get() == 'gpt-3.5-turbo-16k' or api_combo_value.get() == 'GPT-4':
            print('IF IF IF IF IF')
            print('IF IF IF IF IF')
            print('IF IF IF IF IF')
            response_textbox.insert(INSERT, OpenAI_Chat(prompt_textbox.get('1.0', INSERT), True))
        else:
            print('ELSE ELSE ELSE ELSE ELSE')
            print('ELSE ELSE ELSE ELSE ELSE')
            print('ELSE ELSE ELSE ELSE ELSE')
            response_textbox.insert(INSERT, OpenAI_Completion(prompt_textbox.get('1.0', INSERT), True))
        
        print('after if statement')

    #Uncomment this
    #Uncomment this
    # notes_label['text'] = "Begin speaking..."
    # response_textbox.insert(INSERT, o.AskQuestion('OutputFile.wav', 'en-US-Wavenet-J', 'en-US', 'MALE', False, False, response_textbox, INSERT) + '\n', printAllResults)
    #Uncomment this
    #Uncomment this
    
    #response_textbox.insert('insert', AskQuestion('OutputFile.wav', 'en-US-Wavenet-J', 'en-US', 'MALE', False) + '\n', printAllResults)
    # Remove 2nd False and add Radio Button to "Export Audio" or "Don't Export Audio"
    #Also add another Radio button that says "Play Back Response" or "Don't Play Back Response"
    # response = AskQuestion('OutputFile.wav', 'en-US-Wavenet-J', 'en-US', 'MALE')
    # print("printing this stuff: " + response)

#Clear: Sets prompt and text textboxes to blank.
def click_clear_button():
    prompt_textbox.delete('1.0', INSERT)
    response_textbox.delete('1.0', INSERT)

#Copy: Sets prompt textbox equal to text textbox.
def click_copy_button():
    prompt_textbox.delete('1.0', INSERT)
    prompt_textbox.insert('insert', response_textbox.get('1.0', INSERT))
    response_textbox.delete('1.0', INSERT)

#Merge: Sets prompt textbox equal to prompt textbox + '\n' + '\n' + text textbox.
def click_merge_button():
    prompt_textbox.insert('insert', '\n' + '\n' + response_textbox.get('1.0', INSERT))
    response_textbox.delete('1.0', INSERT)

#Reset: Sets all of the parameter textboxes to the default values listed above.
def click_reset_button():
    tokens_textbox.delete('1.0', INSERT)
    temperature_textbox.delete('1.0', INSERT)
    pres_penalty_textbox.delete('1.0', INSERT)
    freq_penalty_textbox.delete('1.0', INSERT)
    best_of_textbox.delete('1.0', INSERT)
    completions_textbox.delete('1.0', INSERT)
    top_p_textbox.delete('1.0', INSERT)
    suffix_textbox.delete('1.0', INSERT)
    stop_1_textbox.delete('1.0', INSERT)
    stop_2_textbox.delete('1.0', INSERT)
    stop_3_textbox.delete('1.0', INSERT)
    stop_4_textbox.delete('1.0', INSERT)
    logit_bias_textbox.delete('1.0', INSERT)
    log_probs_textbox.delete('1.0', INSERT)
    user_textbox.delete('1.0', INSERT)
    category_textbox.delete('1.0', INSERT)
    tags_textbox.delete('1.0', INSERT)
    
    tokens_textbox.insert('insert', '4000')
    temperature_textbox.insert('insert', '0.70')
    pres_penalty_textbox.insert('insert', '0.0')
    freq_penalty_textbox.insert('insert', '0.0')
    best_of_textbox.insert('insert', '1')
    completions_textbox.insert('insert', '1')
    top_p_textbox.insert('insert', '1.00')
    stream_combo.current(1)
    echo_combo.current(1)
    suffix_textbox.insert('insert', '')
    stop_1_textbox.insert('insert', '')
    stop_2_textbox.insert('insert', '')
    stop_3_textbox.insert('insert', '')
    stop_4_textbox.insert('insert', '')
    logit_bias_textbox.insert('insert', '0')
    log_probs_textbox.insert('insert', '0')
    user_textbox.insert('insert', '')
    #title_textbox.insert('insert', 'title')
    #category_textbox.insert('insert', 'cat')
    #tags_textbox.insert('insert', 'tag')
    #tags_textbox.insert('insert', 'Test: ' + str(date.today()))

def Click_Create_One_Page_Button():
    print("Click_Create_One_Page_Button Function")
    #topic = 'Python programming'
    topic = title_textbox.get('1.0', INSERT)
    parent_page = title_textbox.get('1.0', INSERT) #'python'

    current_description = title_textbox.get('1.0', INSERT)
    prompt_textbox.delete('1.0', INSERT)
    prompt_textbox.insert('insert', 'Write a long educational ' + topic + ' article about "' + current_description + '". Include H1, H2, and H3 tags, along with html formatting including bold, underline, italics, font color, pre, etc.')
    
    if prompt_textbox.get('1.0', INSERT) != '':
        if api_combo_value.get() == 'gpt-3.5-turbo':
            openai_code = OpenAI_Chat(prompt_textbox.get('1.0', INSERT), True)
        else:
            openai_code = OpenAI_Completion(prompt_textbox.get('1.0', INSERT), True)

        print('Click_Create_One_Page_Button - prompt_textbox: ' + prompt_textbox.get('1.0', INSERT))
        print('Click_Create_One_Page_Button - topic: ' + topic)
        print('Click_Create_One_Page_Button - parent_page: ' + parent_page)
        print('Click_Create_One_Page_Button - current_description: ' + current_description)
        print('Click_Create_One_Page_Button - openai_code: ' + openai_code)
        p.Upload_WordPress_Page(parent_page, current_description, category_textbox.get('1.0', INSERT), tags_textbox.get('1.0', INSERT), openai_code)

def Click_Create_Outline():
    global inputText
    #prompt_textbox.delete('1.0', INSERT)
    #prompt_textbox.insert('insert', 'Create a comprehensive ' + title_textbox.get('1.0', INSERT) + ' lesson plan outline, covering all essential topics and concepts. Format each section using a pipe (|) delimiter.')

    # inputText = [
    #     {"role": "system", "content": "You are teacher creating a lesson plan for your students."},
    #     {"role": "system", "name": "example_user", "content": "Create a comprehensive Python programming lesson plan outline, covering all essential topics and concepts. Format each section using a pipe (|) delimiter. Don't include any roman numerals, numbers, letters, or bullet points before each value. Include one columns: 'Section'. Don't include any other text before or after the lesson plan; don't even explain your response. Don't include the column name. Include as many sections as possible."},
    #     {"role": "system", "name": "example_assistant", "content": "Introduction to Python|Variables and Data Types|Operators and Expressions|Control Flow|Conditional Statements|Loops (for and while)|Functions|Lists|Tuples|Dictionaries|Sets|String Manipulation|File Handling|Exception Handling|Object-Oriented Programming Basics|Classes and Objects|Inheritance and Polymorphism|Modules and Packages|Working with External Libraries|Regular Expressions|Database Connectivity (SQL)|Web Scraping|GUI Programming (Tkinter)|Introduction to NumPy|Introduction to Pandas|Data Visualization with Matplotlib|Introduction to Flask (Web Framework)|Introduction to Django (Web Framework)|Introduction to Machine Learning with scikit-learn|Introduction to Deep Learning with TensorFlow or PyTorch|Introduction to Natural Language Processing (NLP)|Testing and Debugging|Best Practices and Code Optimization|Deployment and Packaging|Project Work and Application Development|Advanced Topics (optional)|Conclusion and Next Steps"},
    #     {"role": "system", "name": "example_user", "content": "Write a ChatGPT prompt to regenerate the previous response in the exact same format."},
    #     {"role": "system", "name": "example_assistant", "content": "Create a comprehensive Python programming lesson plan outline, covering all essential topics and concepts. Format each section using a pipe (|) delimiter. Don't include any roman numerals, numbers, letters, or bullet points before each value. Include one column: 'Section'. Don't include any other text before or after the lesson plan; don't even explain your response. Don't include the column name. Include as many sections as possible."},
    #     {"role": "system", "name": "example_user", "content": "That prompt doesn't always format the text correctly.  What prompt can I use to always have the text formatted like it was just formatted?"},
    #     {"role": "system", "name": "example_assistant", "content": "Please generate a Python programming lesson plan outline with each section formatted using a pipe (|) delimiter. The lesson plan should cover all essential topics and concepts. The prompt should only include the instruction for formatting the text, without any example or explanation."},
    #     {"role": "system", "name": "example_user", "content": "Please generate a Python programming lesson plan outline with each section formatted using a pipe (|) delimiter. The lesson plan should cover all essential topics and concepts. The prompt should only include the instruction for formatting the text, without any example or explanation."},
    #     {"role": "system", "name": "example_assistant", "content": "Create a comprehensive Python programming lesson plan outline, covering all essential topics and concepts. Format each section using a pipe (|) delimiter."},
    #     {"role": "system", "name": "example_user", "content": "Create a comprehensive Python programming lesson plan outline, covering all essential topics and concepts. Format each section using a pipe (|) delimiter."}
    # ]

    if task_combo.get() == 'Import WordPress Pages':
        inputText = [
            {"role": "system", "content": "You are teacher creating a lesson plan for your students."},
            {"role": "system", "name": "example_user", "content": "Create a comprehensive " + title_textbox.get('1.0', INSERT) + " lesson plan outline, covering all essential topics and concepts. Format each section using a pipe (|) delimiter.  Example: {{Example1}}|{{Example2}}|{{Example3}}|{{Example4}}|{{Example5}}|{{Example6}}|{{Example7}}|{{Example8}}|{{Example9}}|{{Example10}}|{{Example11}}|{{Example12}}|{{Example13}}|{{Example14}}|{{Example15}}|{{Example16}}|{{Example17}}|{{Example18}}|{{Example19}}|{{Example20}}|{{Example21}}|{{Example22}}|{{Example23}}|{{Example24}}|{{Example25}}|{{Example26}}|{{Example27}}|{{Example28}}|{{Example29}}|{{Example30}}|{{Example31}}|{{Example32}}|{{Example33}}|{{Example34}}|{{Example35}}|{{Example36}}|{{Example37}}|{{Example38}}|{{Example39}}|{{Example40}}|{{Example41}}|{{Example42}}|{{Example43}}|{{Example44}}|{{Example45}}|{{Example46}}|{{Example47}}|{{Example48}}|{{Example49}}|{{Example50}}|{{Example51}}|{{Example52}}|{{Example53}}|{{Example54}}|{{Example55}}|{{Example56}}|{{Example57}}|{{Example58}}|{{Example59}}|{{Example60}}|{{Example61}}|{{Example62}}|{{Example63}}|{{Example64}}|{{Example65}}|{{Example66}}|{{Example67}}|{{Example68}}|{{Example69}}|{{Example70}}|{{Example71}}|{{Example72}}|{{Example73}}|{{Example74}}|{{Example75}}|{{Example76}}|{{Example77}}|{{Example78}}|{{Example79}}|{{Example80}}|{{Example81}}|{{Example82}}|{{Example83}}|{{Example84}}|{{Example85}}|{{Example86}}|{{Example87}}|{{Example88}}|{{Example89}}|{{Example90}}|{{Example91}}|{{Example92}}|{{Example93}}|{{Example94}}|{{Example95}}|{{Example96}}|{{Example97}}|{{Example98}}|{{Example99}}|{{Example100}} Replace {{Example1}}, {{Example2}}, and so on, with the actual topics and concepts you want to include in your lesson plan outline. Each section should be enclosed in double curly braces {{}}, and they will be replaced with the respective topics when generating the response. The response will be formatted using pipe (|) delimiters as requested.  Don't include any roman numerals, numbers, letters, or bullet points before each value. Don't include any other text before or after the lesson plan; don't even explain your response."}
            #{"role": "system", "name": "example_user", "content": "Create a comprehensive Python programming lesson plan outline, covering all essential topics and concepts. Format each section using a pipe (|) delimiter.  Example: {{Example1}}|{{Example2}}|{{Example3}}|{{Example4}}|{{Example5}}|{{Example6}}|{{Example7}}|{{Example8}}|{{Example9}}|{{Example10}}|{{Example11}}|{{Example12}}|{{Example13}}|{{Example14}}|{{Example15}}|{{Example16}}|{{Example17}}|{{Example18}}|{{Example19}}|{{Example20}}|{{Example21}}|{{Example22}}|{{Example23}}|{{Example24}}|{{Example25}}|{{Example26}}|{{Example27}}|{{Example28}}|{{Example29}}|{{Example30}}|{{Example31}}|{{Example32}}|{{Example33}}|{{Example34}}|{{Example35}}|{{Example36}}|{{Example37}}|{{Example38}}|{{Example39}}|{{Example40}}|{{Example41}}|{{Example42}}|{{Example43}}|{{Example44}}|{{Example45}}|{{Example46}}|{{Example47}}|{{Example48}}|{{Example49}}|{{Example50}}|{{Example51}}|{{Example52}}|{{Example53}}|{{Example54}}|{{Example55}}|{{Example56}}|{{Example57}}|{{Example58}}|{{Example59}}|{{Example60}}|{{Example61}}|{{Example62}}|{{Example63}}|{{Example64}}|{{Example65}}|{{Example66}}|{{Example67}}|{{Example68}}|{{Example69}}|{{Example70}}|{{Example71}}|{{Example72}}|{{Example73}}|{{Example74}}|{{Example75}}|{{Example76}}|{{Example77}}|{{Example78}}|{{Example79}}|{{Example80}}|{{Example81}}|{{Example82}}|{{Example83}}|{{Example84}}|{{Example85}}|{{Example86}}|{{Example87}}|{{Example88}}|{{Example89}}|{{Example90}}|{{Example91}}|{{Example92}}|{{Example93}}|{{Example94}}|{{Example95}}|{{Example96}}|{{Example97}}|{{Example98}}|{{Example99}}|{{Example100}} Replace {{Example1}}, {{Example2}}, and so on, with the actual topics and concepts you want to include in your lesson plan outline. Each section should be enclosed in double curly braces {{}}, and they will be replaced with the respective topics when generating the response. The response will be formatted using pipe (|) delimiters as requested.  Don't include any roman numerals, numbers, letters, or bullet points before each value. Don't include any other text before or after the lesson plan; don't even explain your response."}
        ]
        prompt_listbox.delete(0, END)
        prompt_list = OpenAI_Chat_Multiple_Messages(inputText)
        inputText = inputText + [{"role": "system", "name": "example_assistant", "content": prompt_list}]
    elif task_combo.get() == 'Write Short Story':
        #prompt = "Write a long and creative outline for 35 chapters of a children's story about " + title_textbox.get('1.0', INSERT) + ". Add many characters. Add a main plot and many sub plots. Add a couple plot twists. Add a lesson at the end. Format each chapter using a pipe (|) delimiter. Here is an example of the format with the pipe symbol to divide the chapters but use your own structure for the outline and chapters insted of using the following structure: Introduction to the planet of Tumble, a world with low gravity, inhabited by very short people called Tumblers.|Meet the main characters: Jumper, Hoops, Dribble, Swish, and Bounce, who form the team 'Gravity Gliders'.|The daily life of Gravity Gliders in Tumble, their love for basketball, and the challenges they face due to their short stature.|Introduction to the rival team, the 'Cosmic Giants', who are taller and stronger.|Gravity Gliders’ struggle to match up to the Giants in the local league.|The arrival of Coach Stardust from another planet, who offers to train the Gravity Gliders.|Training sessions with Coach Stardust, who teaches them to use Tumble's low gravity to their advantage.|Gravity Gliders start winning local matches, gaining confidence and recognition.|Announcement of the Interstellar Basketball Championship, the biggest event in the galaxy.|The Gravity Gliders' rigorous training and preparation for the championship.|The Gravity Gliders' journey to the championship, meeting teams from different planets.|The Gravity Gliders' wins in the initial rounds of the championship, showcasing their improved skills.|The Gravity Gliders make it to the finals, set to face their rivals, the Cosmic Giants.|The suspense builds up as the day of the final match approaches.|The nail-biting final match, where the Gravity Gliders are initially trailing.|Half-time motivation and strategy discussion with Coach Stardust.|The Gravity Gliders make a remarkable comeback in the second half.|Plot Twist: The game ends in a tie and goes into overtime.|The intense overtime period, with both teams giving their all.|The Gravity Gliders win the championship, making them the heroes of Tumble.|The Gravity Gliders' triumphant return to Tumble and the grand celebration.|The Gravity Gliders use their newfound fame to inspire the younger generation in Tumble.|Introduction of a new character, Pogo, a young Tumbler who dreams of joining the Gravity Gliders.|Pogo's journey from being an inexperienced player to a promising talent.|Pogo's training with the Gravity Gliders, learning the ropes of the game.|Pogo's debut in a local match, where he shows his potential.|Pogo officially becomes a part of the Gravity Gliders, marking the beginning of a new era for the team.|The story ends with the lesson that no matter how small you are, with determination and teamwork, you can achieve great heights."
        #prompt = "Write a creative outline for 3 chapters of a children's story about " + title_textbox.get('1.0', INSERT) + ". Add two characters. Add a main plot. Format each chapter using a pipe (|) delimiter. Here is an example of the format with the pipe symbol to divide the chapters but use your own ideas for the outline and chapters insted of copying the following details: A 15-year old boy named Hoops and a 13-year old girl named Dribble are short people living on the planet of Tumble, a world with low gravity, who try out for a basketball team.|Hoops and Dribble don't make the cut for the baseketball team during tryouts, so instead create their own team named 'Gravity Gliders', which includes all short players.|The daily life of Gravity Gliders in Tumble, their love for basketball, and the challenges they face due to their short stature.|Introduction to the rival team, the 'Cosmic Giants', who are taller and stronger.|Gravity Gliders’ struggle to match up to the Giants in the local league.|The arrival of Coach Stardust from another planet, who offers to train the Gravity Gliders.|Training sessions with Coach Stardust, who teaches them to use Tumble's low gravity to their advantage.|Gravity Gliders start winning local matches, gaining confidence and recognition.|Announcement of the Interstellar Basketball Championship, the biggest event in the galaxy.|The Gravity Gliders' rigorous training and preparation for the championship.|The Gravity Gliders' journey to the championship, meeting teams from different planets.|The Gravity Gliders' wins in the initial rounds of the championship, showcasing their improved skills.|The Gravity Gliders make it to the finals, set to face their rivals, the Cosmic Giants.|The suspense builds up as the day of the final match approaches.|The nail-biting final match, where the Gravity Gliders are initially trailing.|Half-time motivation and strategy discussion with Coach Stardust.|The Gravity Gliders make a remarkable comeback in the second half.|The game ends in a tie and goes into overtime.|The intense overtime period, with both teams giving their all.|The Gravity Gliders win the championship, making them the heroes of Tumble.|The Gravity Gliders' triumphant return to Tumble and the grand celebration.|The Gravity Gliders use their newfound fame to inspire the younger generation in Tumble.|"
        prompt = "Write a creative and detailed plot outline for 3 chapters of a children's story about " + title_textbox.get('1.0', INSERT) + ". Add exactly two characters. Add a main plot. Format each chapter using a pipe (|) delimiter. Here is an example of the format with the pipe symbol to divide the chapters but use your own ideas for the outline and chapters insted of copying the following details; also make your plot outline much larger for each of the 3 chapters: A 15-year old boy named Hoops and a 13-year old girl named Dribble are short people living on the planet of Tumble, a world with low gravity, who try out for a basketball team but don't make the cut.|Hoops and Dribble create their own team named 'Gravity Gliders', which includes all short players. They train their new team and prepare for their first game against their rival team the 'Cosmic Giants'.|The Gravity Gliders win their first game against the 'Cosmic Giants' by using the planet's gravity to their advantage to jump higher. The Gravity Gliders use their newfound fame to inspire the younger generation in Tumble.|"
        prompt_textbox.delete('1.0', INSERT)
        prompt_textbox.insert('insert', prompt)
        
        # inputText = [
        #     {"role": "system", "content": "You are a creative writer writing a book."},
        #     {"role": "system", "name": "example_user", "content": prompt}
        # ]

        inputText = prompt
        prompt_listbox.delete(0, END)
        prompt_list = OpenAI_Chat_No_History(inputText, True)
        
    list_items = prompt_list.split("|")
    for item in list_items:
        item_cleaned = item.replace("{", "").replace("}", "")
        prompt_listbox.insert('end', item_cleaned)
    
    # Remove duplicates and update the listbox
    unique_items = list(set(list_items))
    list_items = []

    for item in unique_items:
        list_items.insert(len(list_items), item)

def Click_Continue():
    global inputText
    
    if prompt_textbox.get('1.0', INSERT) != '':
        #inputText = inputText + [{"role": "system", "name": "example_user", "content": "Continue"}]
        inputText = inputText + [{"role": "system", "name": "example_user", "content": prompt_textbox.get('1.0', INSERT)}]
        
        prompt_listbox.delete(0, END)    
        prompt_list = OpenAI_Chat_Multiple_Messages(inputText)
        inputText = inputText + [{"role": "system", "name": "example_assistant", "content": prompt_list}]

        list_items = prompt_list.split("|")
        
        for item in list_items:
            item_cleaned = item.replace("{", "").replace("}", "")
            prompt_listbox.insert('end', item_cleaned)
        
        # Remove duplicates and update the listbox
        unique_items = list(set(list_items))
        list_items = []

        for item in unique_items:
            list_items.insert(END, item)

def Click_Create_Pages_From_List():
    if task_combo.get() == 'Write Short Story':
        # Create an empty list
        chapters = []
        outline = ""

        #Delete story .txt
        filename = r"Short Stories\Story - " + title_textbox.get('1.0', INSERT) + ".txt"
        
        if os.path.exists(filename):
            os.remove(filename)

        for i in range(prompt_listbox.size()):
            outline += prompt_listbox.get(i) + "|"

        outline = outline.rstrip("|")

        # with open("Story - Outline - " + title_textbox.get('1.0', INSERT) + ".txt", "a") as file:
        #     file.write(outline.replace("|", "\n"))
        with open(r"Short Stories\Story - Outline - " + title_textbox.get('1.0', INSERT) + ".txt", "w") as file:
            lines = outline.split("|")
            for i, line in enumerate(lines):
                chapter_number = i + 1
                chapter_line = f"Chapter {chapter_number}: {line.strip()}\n"
                file.write(chapter_line)

        api_combo.current(1) #Set to gpt-3.5-turbo-16k

        for i in range(prompt_listbox.size()):
            if i == 0:
                #prompt = "Write a long and detailed children's story about " + title_textbox.get('1.0', INSERT) + ". Add many characters and character dialogue throughout the story. Make the characters speak to each other. Don't label each page number or chapter number. Only respond with the story. Below is an outline for the story. Only write the story for chapter " + str(i + 1) + " below, which is '" + prompt_listbox.get(i) + "'. I'll ask you to continue after each chapter. " + outline
                prompt = "Write a long and detailed children's story for a 7 year old about " + title_textbox.get('1.0', INSERT) + ". Add two characters and a lot of character dialogue throughout the story. Make the characters speak to each other. Create a plot in chapter 1.  Add a plot twist in chapter 2. Add a lesson at the end of chapter 3. Don't label each page number or chapter number. Only respond with the story. Below is an outline for the story. Only write the story for chapter " + str(i + 1) + " below, which is '" + prompt_listbox.get(i) + "'. I'll ask you to continue after each chapter. " + outline
            else:
                prompt = "Write chapter " + str(i + 1) + " about " + prompt_listbox.get(i)

            print('Prompt(' + str(i) + '): ' + prompt)
            prompt_textbox.delete('1.0', INSERT)
            prompt_textbox.insert('insert', prompt)
            
            if prompt_textbox.get('1.0', INSERT) != '':
                if api_combo_value.get() == 'GPT-3':
                    response = OpenAI_Completion(prompt_textbox.get('1.0', INSERT), True)                
                else:
                    response = OpenAI_Chat(prompt_textbox.get('1.0', INSERT), True)

            print('Response(' + str(i) + '): ' + response)

            with open(r"Short Stories\Story - " + title_textbox.get('1.0', INSERT) + ".txt", "a") as file:
                file.write('\n-----Begin Chapter ' + str(i + 1) + ': ' + prompt_listbox.get(i))
                file.write(response)
                file.write('\n-----End Chapter ' + str(i + 1))

            chapters.append(response)

        #Create character descriptions
        prompt = "List every character in the story from all chapters above. For each character, add a brief description, describing their clothes and look. Don't add filler works such as 'the', 'a', 'and', 'it', etc. Add a pipe symbol after each character name and description. Don't include the word 'chapter'. Don't include chapter numbers. Here is an example but don't copy the details from this example; create your own character descriptions based on the story above: 'Max: 11-year old boy, orange hair, glasses|Bob: monkey, red baseball cap, yellow t-shirt, blue jeans|Kate: 27-year old green alien female, tall skinny, pointy elf ears"
        print('Prompt (Character Descriptions): ' + prompt)
        prompt_textbox.delete('1.0', INSERT)
        prompt_textbox.insert('insert', prompt)
        
        if prompt_textbox.get('1.0', INSERT) != '':
            if api_combo_value.get() == 'GPT-3':
                response = OpenAI_Completion(prompt_textbox.get('1.0', INSERT), True)                
            else:
                response = OpenAI_Chat(prompt_textbox.get('1.0', INSERT), True)

            print('Response (Characters): ' + response)

        with open(r"Short Stories\Story - Characters " + title_textbox.get('1.0', INSERT) + ".txt", "w") as file:
            file.write(response.replace("|", "\n"))

        character_list = response.split("|")
        character_list = [item.strip() for item in character_list if item.strip()]

        #Create setting descriptions
        prompt = "List every setting in the story from all chapters above. For each setting, add a brief description. Don't add filler works such as 'the', 'a', 'and', 'it', etc. Add a pipe symbol after each setting name and description. Don't include the word 'chapter'. Don't include chapter numbers. For example, 'Sandy beach daytime, large purple rocks background|Huge shopping mall, waterfountain escalator background, large ceiling|Large office business meeting, oval glass table, expensive red chairs, glass doors"
        print('Prompt (Setting Descriptions): ' + prompt)
        prompt_textbox.delete('1.0', INSERT)
        prompt_textbox.insert('insert', prompt)
        
        if prompt_textbox.get('1.0', INSERT) != '':
            if api_combo_value.get() == 'GPT-3':
                response = OpenAI_Completion(prompt_textbox.get('1.0', INSERT), True)                
            else:
                response = OpenAI_Chat(prompt_textbox.get('1.0', INSERT), True)

            print('Response (Setting): ' + response)
    
        with open(r"Short Stories\Story - Setting - " + title_textbox.get('1.0', INSERT) + ".txt", "w") as file:
            file.write(response.replace("|", "\n"))

        setting_list = response.split("|")
        setting_list = [item.strip() for item in setting_list if item.strip()]

        #Create MidJourney Prompts
        prompt_listbox.delete(0, END)

        filename = r"Short Stories\Story - Prompts - " + title_textbox.get('1.0', INSERT) + ".txt"
        
        if os.path.exists(filename):
            os.remove(filename)

        for i, chapter in enumerate(chapters):
            print('Response(' + str(i) + '): ' + chapter)

            prompt = "Create 10 MidJourney prompts briefly describing the scene and characters for the story below. Add the character name, verb describing what the character is doing, and the background. Keep prompts brief and to the point. Don't add filler works such as 'the', 'a', 'and', 'it', etc. Don't include any symbols or numbers. Don't include the chapter number. Don't include the word 'chapter'. Add a pipe symbol after each prompt. Here is an example but don't copy the details in this example; create your own prompts from the actual story below: 'Max jumps on red spaceship, expensive neighborhood background|Red spaceship flys into space, yellow sun black space stars background|Portrait view Bob smiling hands in air, expensive office background.'  Here is the story: " + chapter
            print('Prompt (MidJourney  Prompts): ' + prompt)
            prompt_textbox.delete('1.0', INSERT)
            prompt_textbox.insert('insert', prompt)
            
            if prompt_textbox.get('1.0', INSERT) != '':
                if api_combo_value.get() == 'GPT-3':
                    response = OpenAI_Completion(prompt_textbox.get('1.0', INSERT), True)                
                else:
                    response = OpenAI_Chat(prompt_textbox.get('1.0', INSERT), True)

                print('Response (MidJourney Prompts): ' + response)
        
            with open(r"Short Stories\Story - Prompts - " + title_textbox.get('1.0', INSERT) + ".txt", "a") as file:
                file.write('Chapter ' + str(i + 1) + ':\n')
                file.write(response.replace("|", "\n"))

            midjourney_prompt_list = response.split("|")
            midjourney_prompt_list = [item.strip() for item in midjourney_prompt_list if item.strip()]

            for item in midjourney_prompt_list:
                prompt_listbox.insert('end', item)
        
    elif task_combo.get() == 'Import WordPress Pages':
        global summary_content

        last_page = False

        print("Click_Create_Pages_From_List Function")
        #topic = 'Python programming'
        topic = title_textbox.get('1.0', INSERT)
        
        summary_content = "<h1>" + topic + "</h1>"

        for i in range(prompt_listbox.size()):
            print("i: " + str(i))
            if i != 0:
                previous_description = prompt_listbox.get(i - 1)
            else: #First loop through, create a blank summary page (It will eventually contain links for all pages within tutorial)
                #Create a blank page for the user (category) if doesn't already exist
                previous_description = ''
                parent = ''
                title = category_textbox.get('1.0', INSERT)
                print("Right before calling Click_Create_Blank_Page_Button to create the Category page")
                print("parent: " + str(parent))
                print("title: " + title)
                Click_Create_Blank_Page_Button(parent, title)

                print("Right before calling Click_Create_Blank_Page_Button to create the Title page")
                print("parent: " + parent)
                print("title: " + title)
                #Create a blank Summary page for the course
                parent = category_textbox.get('1.0', INSERT) #The parent is the category because this page is the Summary page with links to all pages in course
                title = title_textbox.get('1.0', INSERT)
                Click_Create_Blank_Page_Button(parent, title)

            current_description = prompt_listbox.get(i)
            prompt_textbox.delete('1.0', INSERT)
            prompt_textbox.insert('insert', 'Write a long educational ' + topic + ' article about "' + current_description + '". Include H1, H2, and H3 tags, along with html formatting including bold, underline, italics, font color, pre, etc.')

            if(i != prompt_listbox.size()):
                next_description = prompt_listbox.get(i + 1)
            
            if(i == prompt_listbox.size() - 1):
                last_page = True

            print("last_page: " + str(last_page))
            print("i: " + str(i))
            print("prompt_listbox.size(): " + str(prompt_listbox.size()))
            print("current_description: " + current_description)
            print("next_description: " + next_description)

            url_main = 'https://coderfairy.com'
            parent_page = title_textbox.get('1.0', INSERT)
            #current_url = url_main + '/' + parent_page + '/' + current_description.replace(" ", "-").replace(".", "-").replace("'", "")
            current_url = '/' + current_description.replace(" ", "-").replace(".", "-").replace("'", "")
            summary_content = summary_content + '<a class="page-link" href="' + current_url + '">' + current_description + '</a>'
            #https://coderfairy.com/cat39/title39/page39a/
            Click_Create_Page_Button(previous_description, current_description, next_description)

            #Update the summary page with links to each page (this page should already be created but will be blank)
            if last_page == True:
                title = title_textbox.get('1.0', INSERT)
                page_id = p.Get_Page_ID_By_Title(title)

                new_title = title #"New Page Title"
                print("last_page == True")
                print("page_id: " + str(page_id))
                print("new_title: " + new_title)
                print("summary_content: " + summary_content)
                p.Update_WordPress_Page(page_id, new_title, summary_content)

def Click_Create_Page_Button(previous_description, current_description, next_description):
    print("Click_Create_Page_Button Function")
    if prompt_textbox.get('1.0', INSERT) != '':
        if api_combo_value.get() == 'gpt-3.5-turbo':
            openai_code = OpenAI_Chat_No_History(prompt_textbox.get('1.0', INSERT), True)
        else:
            openai_code = OpenAI_Completion(prompt_textbox.get('1.0', INSERT), True)

        url_main = 'https://coderfairy.com'
        parent_page = title_textbox.get('1.0', INSERT) #'python'

        if previous_description == '':
            previous_description = 'Home'
            previous_url = 'https://coderfairy.com'
        else:
            previous_url = url_main + '/' + parent_page + '/' + previous_description.replace(" ", "-").replace(".", "-").replace("'", "")

        if next_description == '':
            next_description = 'Home'
            next_url = 'https://coderfairy.com'
        else:
            next_url = url_main + '/' + parent_page + '/' + next_description.replace(" ", "-").replace(".", "-").replace("'", "")

        #previous_url = 'https://coderfairy.com/python/1-1-introduction-to-python-course-overview'
        #next_url = 'https://coderfairy.com/code/python/how-to-run-a-python-script-on-wordpress-python-web-tutorial-1/'
        buttons_code = '<div class="button-container"><a class="button" href="' + previous_url + '">&lt; Previous</a><br /><a class="button" href="' + next_url + '">Next &gt;</a></div><br><br><br>'

        button_descriptions_code = '<div class="button-container"><a href="' + previous_url + '">' + previous_description + '</a><br><a href="' + next_url + '">' + next_description + '</a></div>'

        content = buttons_code + ' ' + openai_code + buttons_code + button_descriptions_code
        #p.Upload_WordPress_Page(parent_page, current_description, content)
        #p.Upload_WordPress_Page(parent_page, current_description, category_textbox.get('1.0', INSERT), tags_textbox.get('1.0', INSERT), openai_code)
        p.Upload_WordPress_Page(parent_page, current_description, category_textbox.get('1.0', INSERT), tags_textbox.get('1.0', INSERT), content)
        #p.Upload_WordPress_Page(parent_page, title_textbox.get('1.0', INSERT), content)

        # response_textbox.delete('1.0', INSERT)
        #     response_textbox.insert(INSERT, OpenAI_Chat(prompt_textbox.get('1.0', INSERT), True))
        #     response_textbox.insert(INSERT, OpenAI_Completion(prompt_textbox.get('1.0', INSERT), True))
        #Write a long article about "Conditional statements (if/else)" for Lesson 2.  Include H1, H2, and H3 tags, along with html formatting including bold, underline, italics, font color, etc.
        #1.4 - Introduction to Python - Variables and Data Types
        #Upload_WordPress_Page('Python', '000 - This is My test Title!', 'Test content')

def Click_Create_Blank_Page_Button(parent, title):
    print("Click_Create_Blank_Page_Button Function")
    url = 'https://coderfairy.com/wp-json/wp/v2'
    user = 'BrandonPodell'
    password = 'BFUp 9tzN W0AE f6hK E1kf 97lx'
    auth_credentials = (user, password)

    category_url = url + '/categories' #'https://coderfairy.com/wp-json/wp/v2/categories'
    tag_url = url + '/tags' #'https://coderfairy.com/wp-json/wp/v2/tags'

    category = category_textbox.get('1.0', INSERT)
    tag = tags_textbox.get('1.0', INSERT)
    content = ''
    print('title: ' + title)
    print('parent: ' + str(parent))
    print('category: ' + category)
    print('tag: ' + tag)
    #parent = 'python'
    p.create_wordpress_category(category, category_url, auth_credentials)
    p.create_wordpress_tag(tag, tag_url, auth_credentials)
    p.Upload_WordPress_Page(parent, title, category, tag, content)

def Click_create_prompt_button():
    prompt_textbox.delete('1.0', INSERT)
    prompt_textbox.insert('insert', 'Write a long educational article about "' + title_textbox.get('1.0', INSERT) + '". Include H1, H2, and H3 tags, along with html formatting including bold, underline, italics, pre, etc.')

# Create form
form = Tk()
form.state('zoomed')
form.title('www.coderfairy.com')
form.config(bg = bgcolor)

largefont = TkFont.Font(family="Helvetica",size=14)
form.option_add("*TCombobox*Font", largefont)
#form.option_add("*TCombobox*Listbox*Font", largefont)
#form.option_add("*Font", largefont) #Changes font for all combo boxes

type_combo_value = tkr.StringVar()
type_combo = tkrttk.Combobox(form, textvariable = type_combo_value)
type_combo.config(state='readonly')
type_combo['values'] = ['Text', 'Image', 'Video', 'Audio']
type_combo.current(0)
type_combo.bind("<<ComboboxSelected>>", click_type_combo)
type_combo.pack()
form.update()
type_combo.place(x = padding, y = padding, width = 110, height = 30)
#type_combo_value.set("hello") #Changes combobox's current value
#print(type_combo_value.get())

api_combo_value = tkr.StringVar()
api_combo = tkrttk.Combobox(form, textvariable = api_combo_value)
api_combo.config(state='readonly')
api_combo['values'] = ['gpt-4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo', 'GPT-3']
api_combo.current(0)
api_combo.pack()
form.update()
api_combo.place(x = padding + type_combo.winfo_x() + type_combo.winfo_width(), y = type_combo.winfo_y(), width = 150, height = type_combo.winfo_height())
#api_combo_value.set("hello") #Changes combobox's current value
#print(api_combo_value.get())

#form.option_add("*TCombobox*Listbox*Font", largefont)
model_combo_value = tkr.StringVar()
model_combo = tkrttk.Combobox(form, textvariable = model_combo_value)
model_combo.config(state='readonly')
model_combo['values'] = ['text-davinci-003' , 'text-curie-001', 'text-babbage-001', 'text-ada-001']
model_combo.current(3)
model_combo.pack()
form.update()
model_combo.place(x = padding + api_combo.winfo_x() + api_combo.winfo_width(), y = api_combo.winfo_y(), width = 175, height = api_combo.winfo_height())

task_combo_value = tkr.StringVar()
task_combo = tkrttk.Combobox(form, textvariable = task_combo_value)
task_combo.config(state='readonly')
task_combo['values'] = ['Write Short Story' , 'Import WordPress Pages']
task_combo.current(0)
task_combo.pack()
form.update()
task_combo.place(x = padding + model_combo.winfo_x() + model_combo.winfo_width(), y = model_combo.winfo_y(), width = 175, height = model_combo.winfo_height())

# save_api_key_button = Button(form, text = "Save API Key", command = Click_save_api_key_button)
# save_api_key_button.pack()
# form.update()
# save_api_key_button.place(x = padding + model_combo.winfo_x() + model_combo.winfo_width(), y = model_combo.winfo_y(), width = 100, height = model_combo.winfo_height())

create_prompt_button = Button(form, text = "Create Prompt", command = Click_create_prompt_button)
create_prompt_button.pack()
form.update()
create_prompt_button.place(x = padding + task_combo.winfo_x() + task_combo.winfo_width(), y = task_combo.winfo_y(), width = 100, height = task_combo.winfo_height())

title_textbox = ''
title_textbox = Text(form)
title_textbox.pack()
form.update()
title_textbox.configure(font = textbox_font)
title_textbox.place(x = padding + create_prompt_button.winfo_x() + create_prompt_button.winfo_width(), y = create_prompt_button.winfo_y(), width = 300, height = create_prompt_button.winfo_height())

category_textbox = ''
category_textbox = Text(form)
category_textbox.pack()
form.update()
category_textbox.configure(font = textbox_font)
category_textbox.place(x = padding + title_textbox.winfo_x() + title_textbox.winfo_width(), y = title_textbox.winfo_y(), width = 130, height = title_textbox.winfo_height())

tags_textbox = ''
tags_textbox = Text(form)
tags_textbox.pack()
form.update()
tags_textbox.configure(font = textbox_font)
tags_textbox.place(x = padding + category_textbox.winfo_x() + category_textbox.winfo_width(), y = category_textbox.winfo_y(), width = 130, height = category_textbox.winfo_height())

# notes_label = Label(form, text = "Click the Submit button to ask the Coder Fairy a question.", anchor = 'w')
# notes_label.pack()
# form.update()
# notes_label.place(x = save_api_key_button.winfo_x() + save_api_key_button.winfo_width() + padding, y = save_api_key_button.winfo_y(), width = 500, height = save_api_key_button.winfo_height())

prompt_text = ''
prompt_textbox = Text(form)
prompt_textbox.pack(expand = True)
#prompt_textbox.insert('insert', prompt_textbox)
#text_box.config(state='disabled')
form.update()
prompt_textbox.place(x = type_combo.winfo_x(), y = type_combo.winfo_y() + type_combo.winfo_height() + padding, width = form.winfo_width() - 100 - width_of_labels - padding * 4 - width_of_listbox - padding * 2, height = 300)

submit_button = Button(form, text = "Submit", command = Click_submit_button)
submit_button.pack()
form.update()
submit_button.place(x = prompt_textbox.winfo_x(), y = form.winfo_height() - 40, width = button_width, height = button_height)

clear_button = Button(form, text = "Clear", command =click_clear_button)
clear_button.pack()
form.update()
clear_button.place(x = submit_button.winfo_x() + submit_button.winfo_width() + padding, y = submit_button.winfo_y(), width = submit_button.winfo_width(), height = submit_button.winfo_height())

copy_button = Button(form, text = "Copy", command = click_copy_button)
copy_button.pack()
form.update()
copy_button.place(x = clear_button.winfo_x() + clear_button.winfo_width() + padding, y = clear_button.winfo_y(), width = clear_button.winfo_width(), height = clear_button.winfo_height())

merge_button = Button(form, text = "Merge", command = click_merge_button)
merge_button.pack()
form.update()
merge_button.place(x = copy_button.winfo_x() + copy_button.winfo_width() + padding, y = copy_button.winfo_y(), width = copy_button.winfo_width(), height = copy_button.winfo_height())

reset_button = Button(form, text = "Reset", command = click_reset_button)
reset_button.pack()
form.update()
reset_button.place(x = merge_button.winfo_x() + merge_button.winfo_width() + padding, y = merge_button.winfo_y(), width = merge_button.winfo_width(), height = merge_button.winfo_height())

generate_prompt_list_button = Button(form, text = "Create Prompts", command = Click_Generate_Prompt_List)
generate_prompt_list_button.pack()
form.update()
generate_prompt_list_button.place(x = reset_button.winfo_x() + reset_button.winfo_width() + padding, y = reset_button.winfo_y(), width = reset_button.winfo_width(), height = reset_button.winfo_height())

create_page_button = Button(form, text = "Create Page", command = Click_Create_One_Page_Button)
create_page_button.pack()
form.update()
create_page_button.place(x = generate_prompt_list_button.winfo_x() + generate_prompt_list_button.winfo_width() + padding, y = generate_prompt_list_button.winfo_y(), width = generate_prompt_list_button.winfo_width(), height = generate_prompt_list_button.winfo_height())

create_pages_from_list_button = Button(form, text = "Create All Pages", command = Click_Create_Pages_From_List)
create_pages_from_list_button.pack()
form.update()
create_pages_from_list_button.place(x = create_page_button.winfo_x() + create_page_button.winfo_width() + padding, y = create_page_button.winfo_y(), width = create_page_button.winfo_width(), height = create_page_button.winfo_height())

create_outline_button = Button(form, text = "Create Outline", command = Click_Create_Outline)
create_outline_button.pack()
form.update()
create_outline_button.place(x = create_pages_from_list_button.winfo_x() + create_pages_from_list_button.winfo_width() + padding, y = create_pages_from_list_button.winfo_y(), width = create_pages_from_list_button.winfo_width(), height = create_pages_from_list_button.winfo_height())

continue_button = Button(form, text = "Continue", command = Click_Continue)
continue_button.pack()
form.update()
continue_button.place(x = create_outline_button.winfo_x() + create_outline_button.winfo_width() + padding, y = create_outline_button.winfo_y(), width = create_outline_button.winfo_width(), height = create_outline_button.winfo_height())




response_textbox = ''
response_textbox = Text(form)
response_textbox.pack(expand = True)
form.update()
response_textbox.place(x = prompt_textbox.winfo_x(), y = prompt_textbox.winfo_y() + prompt_textbox.winfo_height() + padding, width = prompt_textbox.winfo_width(), height = submit_button.winfo_y() - response_textbox.winfo_height() - padding)

#Create textboxes for GPT-3's settings in the last column (all the way to the right of the screen)
tokens_textbox = ''
tokens_textbox = Text(form)
tokens_textbox.pack()
form.update()
tokens_textbox.configure(font = textbox_font)
tokens_textbox.place(x = form.winfo_width() - 100 - padding, y = prompt_textbox.winfo_y(), width = 100, height = type_combo.winfo_height())

temperature_textbox = ''
temperature_textbox = Text(form)
temperature_textbox.pack()
form.update()
temperature_textbox.configure(font = textbox_font)
temperature_textbox.place(x = tokens_textbox.winfo_x(), y = tokens_textbox.winfo_y() + tokens_textbox.winfo_height() + padding2, width = tokens_textbox.winfo_width(), height = tokens_textbox.winfo_height())

pres_penalty_textbox = ''
pres_penalty_textbox = Text(form)
pres_penalty_textbox.pack()
form.update()
pres_penalty_textbox.configure(font = textbox_font)
pres_penalty_textbox.place(x = temperature_textbox.winfo_x(), y = temperature_textbox.winfo_y() + temperature_textbox.winfo_height() + padding2, width = temperature_textbox.winfo_width(), height = temperature_textbox.winfo_height())

freq_penalty_textbox = ''
freq_penalty_textbox = Text(form)
freq_penalty_textbox.pack()
form.update()
freq_penalty_textbox.configure(font = textbox_font)
freq_penalty_textbox.place(x = pres_penalty_textbox.winfo_x(), y = pres_penalty_textbox.winfo_y() + pres_penalty_textbox.winfo_height() + padding2, width = pres_penalty_textbox.winfo_width(), height = pres_penalty_textbox.winfo_height())

best_of_textbox = ''
best_of_textbox = Text(form)
best_of_textbox.pack()
form.update()
best_of_textbox.configure(font = textbox_font)
best_of_textbox.place(x = pres_penalty_textbox.winfo_x(), y = pres_penalty_textbox.winfo_y() + pres_penalty_textbox.winfo_height() + padding2, width = pres_penalty_textbox.winfo_width(), height = pres_penalty_textbox.winfo_height())

completions_textbox = ''
completions_textbox = Text(form)
completions_textbox.pack()
form.update()
completions_textbox.configure(font = textbox_font)
completions_textbox.place(x = best_of_textbox.winfo_x(), y = best_of_textbox.winfo_y() + best_of_textbox.winfo_height() + padding2, width = best_of_textbox.winfo_width(), height = best_of_textbox.winfo_height())

top_p_textbox = ''
top_p_textbox = Text(form)
top_p_textbox.pack()
form.update()
top_p_textbox.configure(font = textbox_font)
top_p_textbox.place(x = completions_textbox.winfo_x(), y = completions_textbox.winfo_y() + completions_textbox.winfo_height() + padding2, width = completions_textbox.winfo_width(), height = completions_textbox.winfo_height())

stream_combo_value = tkr.StringVar()
stream_combo = tkrttk.Combobox(form, textvariable = stream_combo_value)
stream_combo.config(state='readonly')
stream_combo['values'] = ['True', 'False']
stream_combo.pack()
form.update()
stream_combo.place(x = top_p_textbox.winfo_x(), y = top_p_textbox.winfo_y() + top_p_textbox.winfo_height() + padding2, width = top_p_textbox.winfo_width(), height = top_p_textbox.winfo_height())

echo_combo_value = tkr.StringVar()
echo_combo = tkrttk.Combobox(form, textvariable = echo_combo_value)
echo_combo.config(state='readonly')
echo_combo['values'] = ['True', 'False']
echo_combo.pack()
form.update()
echo_combo.place(x = stream_combo.winfo_x(), y = stream_combo.winfo_y() + stream_combo.winfo_height() + padding2, width = stream_combo.winfo_width(), height = stream_combo.winfo_height())

suffix_textbox = ''
suffix_textbox = Text(form)
suffix_textbox.pack()
form.update()
suffix_textbox.configure(font = textbox_font)
suffix_textbox.place(x = echo_combo.winfo_x(), y = echo_combo.winfo_y() + echo_combo.winfo_height() + padding2, width = echo_combo.winfo_width(), height = echo_combo.winfo_height())

stop_1_textbox = ''
stop_1_textbox = Text(form)
stop_1_textbox.pack()
form.update()
stop_1_textbox.configure(font = textbox_font)
stop_1_textbox.place(x = suffix_textbox.winfo_x(), y = suffix_textbox.winfo_y() + suffix_textbox.winfo_height() + padding2, width = suffix_textbox.winfo_width(), height = suffix_textbox.winfo_height())

stop_2_textbox = ''
stop_2_textbox = Text(form)
stop_2_textbox.pack()
form.update()
stop_2_textbox.configure(font = textbox_font)
stop_2_textbox.place(x = stop_1_textbox.winfo_x(), y = stop_1_textbox.winfo_y() + stop_1_textbox.winfo_height() + padding2, width = stop_1_textbox.winfo_width(), height = stop_1_textbox.winfo_height())

stop_3_textbox = ''
stop_3_textbox = Text(form)
stop_3_textbox.pack()
form.update()
stop_3_textbox.configure(font = textbox_font)
stop_3_textbox.place(x = stop_2_textbox.winfo_x(), y = stop_2_textbox.winfo_y() + stop_2_textbox.winfo_height() + padding2, width = stop_2_textbox.winfo_width(), height = stop_2_textbox.winfo_height())

stop_4_textbox = ''
stop_4_textbox = Text(form)
stop_4_textbox.pack()
form.update()
stop_4_textbox.configure(font = textbox_font)
stop_4_textbox.place(x = stop_3_textbox.winfo_x(), y = stop_3_textbox.winfo_y() + stop_3_textbox.winfo_height() + padding2, width = stop_3_textbox.winfo_width(), height = stop_3_textbox.winfo_height())

logit_bias_textbox = ''
logit_bias_textbox = Text(form)
logit_bias_textbox.pack()
form.update()
logit_bias_textbox.configure(font = textbox_font)
logit_bias_textbox.place(x = stop_4_textbox.winfo_x(), y = stop_4_textbox.winfo_y() + stop_4_textbox.winfo_height() + padding2, width = stop_4_textbox.winfo_width(), height = stop_4_textbox.winfo_height())

log_probs_textbox = ''
log_probs_textbox = Text(form)
log_probs_textbox.pack()
form.update()
log_probs_textbox.configure(font = textbox_font)
log_probs_textbox.place(x = logit_bias_textbox.winfo_x(), y = logit_bias_textbox.winfo_y() + logit_bias_textbox.winfo_height() + padding2, width = logit_bias_textbox.winfo_width(), height = logit_bias_textbox.winfo_height())

user_textbox = ''
user_textbox = Text(form)
user_textbox.pack()
form.update()
user_textbox.configure(font = textbox_font)
user_textbox.place(x = log_probs_textbox.winfo_x(), y = log_probs_textbox.winfo_y() + log_probs_textbox.winfo_height() + padding2, width = log_probs_textbox.winfo_width(), height = log_probs_textbox.winfo_height())

#Create labels in 2nd to last column (to the left of the textboxes)
tokens_label = Label(form, text = "Tokens", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
tokens_label.pack()
form.update()
tokens_label.place(x = tokens_textbox.winfo_x() - width_of_labels - padding, y = tokens_textbox.winfo_y(), width = width_of_labels, height = height_of_labels)

temperature_label = Label(form, text = "Temperature", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
temperature_label.pack()
form.update()
temperature_label.place(x = tokens_label.winfo_x(), y = tokens_label.winfo_y() + tokens_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

pres_penalty_label = Label(form, text = "Pres. Penalty", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
pres_penalty_label.pack()
form.update()
pres_penalty_label.place(x = temperature_label.winfo_x(), y = temperature_label.winfo_y() + temperature_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

freq_penalty_label = Label(form, text = "Freq. Penalty", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
freq_penalty_label.pack()
form.update()
freq_penalty_label.place(x = pres_penalty_label.winfo_x(), y = pres_penalty_label.winfo_y() + pres_penalty_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

best_of_label = Label(form, text = "Best of", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
best_of_label.pack()
form.update()
best_of_label.place(x = pres_penalty_label.winfo_x(), y = pres_penalty_label.winfo_y() + pres_penalty_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

completions_label = Label(form, text = "Completions", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
completions_label.pack()
form.update()
completions_label.place(x = best_of_label.winfo_x(), y = best_of_label.winfo_y() + best_of_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

top_p_label = Label(form, text = "Top P", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
top_p_label.pack()
form.update()
top_p_label.place(x = completions_label.winfo_x(), y = completions_label.winfo_y() + completions_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

stream_label = Label(form, text = "Stream", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
stream_label.pack()
form.update()
stream_label.place(x = top_p_label.winfo_x(), y = top_p_label.winfo_y() + top_p_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

echo_label = Label(form, text = "Echo", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
echo_label.pack()
form.update()
echo_label.place(x = stream_label.winfo_x(), y = stream_label.winfo_y() + stream_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

suffix_label = Label(form, text = "Suffix", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
suffix_label.pack()
form.update()
suffix_label.place(x = echo_label.winfo_x(), y = echo_label.winfo_y() + echo_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

stop_1_label = Label(form, text = "Stop 1", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
stop_1_label.pack()
form.update()
stop_1_label.place(x = suffix_label.winfo_x(), y = suffix_label.winfo_y() + suffix_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

stop_2_label = Label(form, text = "Stop 2", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
stop_2_label.pack()
form.update()
stop_2_label.place(x = stop_1_label.winfo_x(), y = stop_1_label.winfo_y() + stop_1_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

stop_3_label = Label(form, text = "Stop 3", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
stop_3_label.pack()
form.update()
stop_3_label.place(x = stop_2_label.winfo_x(), y = stop_2_label.winfo_y() + stop_2_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

stop_4_label = Label(form, text = "Stop 4", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
stop_4_label.pack()
form.update()
stop_4_label.place(x = stop_3_label.winfo_x(), y = stop_3_label.winfo_y() + stop_3_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

logit_bias_label = Label(form, text = "Logit Bias", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
logit_bias_label.pack()
form.update()
logit_bias_label.place(x = stop_4_label.winfo_x(), y = stop_4_label.winfo_y() + stop_4_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

log_probs_label = Label(form, text = "Lag Prob.", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
log_probs_label.pack()
form.update()
log_probs_label.place(x = logit_bias_label.winfo_x(), y = logit_bias_label.winfo_y() + logit_bias_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

user_label = Label(form, text = "User", bg = bgcolor, fg = fgcolor, font = label_font_type, anchor="e")
user_label.pack()
form.update()
user_label.place(x = log_probs_label.winfo_x(), y = log_probs_label.winfo_y() + log_probs_label.winfo_height() + padding2, width = width_of_labels, height = height_of_labels)

cost_label = Label(form, text = "Cost", bg = bgcolor, fg = fgcolor, font = label_font_type2, anchor="e")
cost_label.pack()
form.update()
cost_label.place(x = tokens_label.winfo_x(), y = padding2, width = width_of_labels)

total_cost_label = Label(form, text = "Total Cost", bg = bgcolor, fg = fgcolor, font = label_font_type2, anchor="e")
total_cost_label.pack()
form.update()
total_cost_label.place(x = cost_label.winfo_x(), y = cost_label.winfo_y() + cost_label.winfo_height(), width = width_of_labels)

cost_value_label = Label(form, text = "$0.00", bg = bgcolor, fg = fgcolor, font = label_font_type2, anchor="w")
cost_value_label.pack()
form.update()
cost_value_label.place(x = tokens_textbox.winfo_x(), y = padding2, width = width_of_labels)

total_cost_value_label = Label(form, text = "$0.00", bg = bgcolor, fg = fgcolor, font = label_font_type2, anchor="w")
total_cost_value_label.pack()
form.update()
total_cost_value_label.place(x = cost_value_label.winfo_x(), y = cost_value_label.winfo_y() + cost_value_label.winfo_height(), width = width_of_labels)

var2 = StringVar()
#var2.set((1,2,3,4))
prompt_listbox = Listbox(form, listvariable=var2)


# function to delete the selected item
def delete_item(event):
    selected_index = prompt_listbox.curselection()[0]
    prompt_listbox.delete(selected_index)
    prompt_listbox.selection_set(selected_index-1)

    if selected_index == 0:
        prompt_listbox.selection_set(0)

# function to move the selected item up
def move_up(event):
    selected_index = prompt_listbox.curselection()[0]
    if selected_index > 0:
        item = prompt_listbox.get(selected_index)
        prompt_listbox.delete(selected_index)
        prompt_listbox.insert(selected_index-1, item)
        prompt_listbox.selection_clear(0, END)
        prompt_listbox.selection_set(selected_index-1)
    # selected_index = prompt_listbox.curselection()[0]
    # if selected_index > 0:
    #     prompt_listbox.selection_clear(0, END)
    #     prompt_listbox.selection_set(selected_index-1)
        prompt_listbox.activate(selected_index)

# function to move the selected item down
def move_down(event):
    selected_index = prompt_listbox.curselection()[0]
    if selected_index < prompt_listbox.size()-1:
        item = prompt_listbox.get(selected_index)
        prompt_listbox.delete(selected_index)
        prompt_listbox.insert(selected_index+1, item)
        prompt_listbox.selection_clear(0, END)
        prompt_listbox.selection_set(selected_index+1)

# function to create a new item
# def create_item(event):
#     selected_index = prompt_listbox.curselection()[0]
#     item = prompt_listbox.get(selected_index)
#     new_item = simpledialog.askstring("Create item", "Enter a new item:")
#     if new_item:
#         prompt_listbox.insert(selected_index+1, new_item)

# function to create a new item
def create_item(event):
    if prompt_listbox.size() == 0:
        new_item = simpledialog.askstring("Create item", "Enter a new item:")
        if new_item:
            prompt_listbox.insert(END, new_item)
    else:
        selected_index = prompt_listbox.curselection()
        if selected_index:
            selected_index = selected_index[0]
            item = prompt_listbox.get(selected_index)
            new_item = simpledialog.askstring("Create item", "Enter a new item:")
            if new_item:
                prompt_listbox.insert(selected_index+1, new_item)
                
# bind events to the listbox widget
prompt_listbox.bind("<Delete>", delete_item)
prompt_listbox.bind("<Up>", move_up)
prompt_listbox.bind("<Down>", move_down)
prompt_listbox.bind("<Key>", create_item)
prompt_listbox.bind("<Button-1>", create_item)




prompt_listbox.pack()
prompt_listbox.place(x = prompt_textbox.winfo_x() + prompt_textbox.winfo_width() + padding, y = prompt_textbox.winfo_y(), width = width_of_listbox, height = prompt_textbox.winfo_height() + response_textbox.winfo_height() + padding)

click_reset_button()

# promptType = IntVar()
# prompt_type_label = Label(form, text = 'Prompt Type', padx = 20)
# prompt_type_label.pack()
# form.update()
# prompt_type_label.place(x = submit_button.winfo_x() + submit_button.winfo_width() + 10, y = submit_button.winfo_y())

# continue_radiobutton = Radiobutton(form, text = "Continue", padx = 20, variable = promptType, value = 1)
# new_radiobutton = Radiobutton(form, text = "New", padx = 20, variable = promptType, value = 2)
# continue_radiobutton.pack(anchor = W)
# new_radiobutton.pack(anchor = W)
# form.update()
# continue_radiobutton.place(x = prompt_type_label.winfo_x(), y = prompt_type_label.winfo_y() + continue_radiobutton.winfo_height())
# form.update()
# new_radiobutton.place(x = continue_radiobutton.winfo_x() + continue_radiobutton.winfo_width(), y = continue_radiobutton.winfo_y())
# continue_radiobutton.select()

# includeType = IntVar()
# include_question_and_answer_radiobutton = Radiobutton(form, text = "Include Question & Answer", padx = 20, variable = includeType, value = 1)
# include_answer_radiobutton = Radiobutton(form, text = "Include Answer Only", padx = 20, variable = includeType, value = 2)
# include_question_and_answer_radiobutton.pack(anchor = W)
# include_answer_radiobutton.pack(anchor = W)
# form.update()
# include_question_and_answer_radiobutton.place(x = continue_radiobutton.winfo_x(), y = continue_radiobutton.winfo_y() + include_question_and_answer_radiobutton.winfo_height())
# form.update()
# include_answer_radiobutton.place(x = include_question_and_answer_radiobutton.winfo_x() + include_question_and_answer_radiobutton.winfo_width(), y = include_question_and_answer_radiobutton.winfo_y())
# include_question_and_answer_radiobutton.select()

# modeType = IntVar()
# completion_mode_radiobutton = Radiobutton(form, text = "Completion Mode", padx = 20, variable = modeType, value = 1)
# store_questions_radiobutton = Radiobutton(form, text = "Store Questions", padx = 20, variable = modeType, value = 2)
# completion_mode_radiobutton.pack(anchor = W)
# store_questions_radiobutton.pack(anchor = W)
# form.update()
# completion_mode_radiobutton.place(x = continue_radiobutton.winfo_x(), y = include_question_and_answer_radiobutton.winfo_y() + completion_mode_radiobutton.winfo_height())
# form.update()
# store_questions_radiobutton.place(x = completion_mode_radiobutton.winfo_x() + completion_mode_radiobutton.winfo_width(), y = completion_mode_radiobutton.winfo_y())
# completion_mode_radiobutton.select()

form.mainloop()




#Create a menu
# def donothing():
#    filewin = Toplevel(form)
#    button = Button(filewin, text="Do nothing button")
#    button.pack()
   
# menubar = Menu(form)
# filemenu = Menu(menubar, tearoff=0)
# filemenu.add_command(label="New", command=donothing)
# filemenu.add_command(label="Open", command=donothing)
# filemenu.add_command(label="Save", command=donothing)
# filemenu.add_command(label="Save as...", command=donothing)
# filemenu.add_command(label="Close", command=donothing)

# filemenu.add_separator()

# filemenu.add_command(label="Exit", command=form.quit)
# menubar.add_cascade(label="File", menu=filemenu)
# editmenu = Menu(menubar, tearoff=0)
# editmenu.add_command(label="Undo", command=donothing)

# editmenu.add_separator()

# editmenu.add_command(label="Cut", command=donothing)
# editmenu.add_command(label="Copy", command=donothing)
# editmenu.add_command(label="Paste", command=donothing)
# editmenu.add_command(label="Delete", command=donothing)
# editmenu.add_command(label="Select All", command=donothing)

# menubar.add_cascade(label="Edit", menu=editmenu)
# helpmenu = Menu(menubar, tearoff=0)
# helpmenu.add_command(label="Help Index", command=donothing)
# helpmenu.add_command(label="About...", command=donothing)
# menubar.add_cascade(label="Help", menu=helpmenu)

# form.config(menu=menubar)


x = 5
y = 3

print(not x == y) # Output: True