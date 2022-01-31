from tkinter import *
from utils import *

model = load_model()
model.eval()

#### This function will be call when we will press the "Send Tweet to Model".

def method_fetch_given_Text():
    class_dict = {0: 'Offensive', 1: 'Not Offensive'}
    ### We will get the tweet from text box and apply pre processing function and send to the model for prediction.
    text_ = text.get("1.0", END)
    print(len(text_))
    # Checking lenth
    if 1 < len(text_) < 10:
        label_last['text'] = 'Your text input length is small for BERT evaluation!'
    elif len(text_) <= 1:
        label_last['text'] = 'Input field is empty!'
    else:
        text_ = preprocess(text_)
        encoding = get_encoding(text_)
        result = class_dict[get_prediction(encoding, model)]

        if len(text_) > 70:
            text_ = text_[:70] + '\n' + text_[70:]
        ### priting results
        label_last['text'] = 'The sentence is :' + '\n' + str(text_) + '\n\n' + str(result)


#### This function will be call when we will press the "Random Tweet Fetch".
def method_fetch_random_Text():
    text_random = center_frame_text.get(1.0, END)
    ### call the random fetch tweet function
    text_ = get_random_tweet_from_tweepy(text_random)
    ### process it
    text_ = preprocess(text_)
    class_dict = {0: 'Offensive', 1: 'Not Offensive'}
    # print(text_)
    # get the endcoding
    encoding = get_encoding(text_)
    ### get the results
    result = class_dict[get_prediction(encoding, model)]


    if len(text_) > 70:
        text_ = text_[:70] + '\n' + text_[70:]
    if len(text_) <= 1:
        label_last['text'] = 'Fetched Tweet length is small !'
    else:
        label_last['text'] = 'The sentence is :' + '\n' + str(text_) + '\n\n' + str(result)


root = Tk()
root.title("Bert Hate Synthesis")
frame = Frame(root)
root.geometry("600x400")
root.resizable(width=False, height=False)
frame.pack()

topFrame = Frame(root, padx=10, pady=10)
l1 = Label(topFrame, text="Bert Hate Speech \n Semantic Segmentation")
l1.pack()
topFrame.pack()

top_left_frame = Frame(root, borderwidth=1, relief=RIDGE, padx=10, pady=10)
top_left_frame.pack()
var = StringVar()
label = Label(top_left_frame, textvariable=var)
var.set("Tweet : ")
label.pack(side=TOP, anchor=NW)
text = Text(top_left_frame, width=40, height=2, padx=10)
text.pack(side=LEFT, anchor=NW)

blackbutton = Button(top_left_frame, text="Send Tweet to Model", fg="black", padx=5, command=method_fetch_given_Text)
blackbutton.pack(side=RIGHT, anchor=NE)

center_frame = Frame(root, pady=20)
center_frame.pack()
random_fetch_button = Button(center_frame, text="Random Tweet Fetch", fg="black",  command=method_fetch_random_Text)
random_fetch_button.pack(side=RIGHT)
var2 = StringVar()
center_frame_label = Label(center_frame, textvariable = var2)
var2.set('Write Random Fetch Keyword: ')
center_frame_label.pack(side=TOP, anchor=NW)

center_frame_text = Text(center_frame, width=40, height=2,)
center_frame_text.pack(side=LEFT, anchor=SW)

bottom_Frame = Frame(root, borderwidth=1, relief=RIDGE, padx=19, pady=4)
bottom_Frame.pack()
label_last = Label(bottom_Frame, text="The result Text.")
label_last.pack()

root.mainloop()
