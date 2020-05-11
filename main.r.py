import os
import tkinter.messagebox
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import asksaveasfilename
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import requests
import nltk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from itertools import islice
from wordcloud import WordCloud, STOPWORDS
from keras import backend as be
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from bs4 import BeautifulSoup

nltk.download('stopwords')

# MAIN EMPTY WINDOW
class Win:
    def __init__(self, root):
        self.root = root
        self.add_menu()
        self.root.geometry("400x300")

    def add_menu(self):
        self.menubar = Menu(self.root, tearoff=False)

        self.menubar.add_command(label="Scrap", command=self.scrap)

        create = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="Create", menu=create)

        create.add_command(label="Neural Network Model", command=self.create_model)

        analyze = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="Analyze", menu=analyze)

        analyze.add_command(label="Neural Network", command=self.neural_network)
        analyze.add_command(label="Naive Bayes", command=self.naive_bayes)
        analyze.add_command(label="Dictionary", command=self.dictionary)
        analyze.add_separator()
        analyze.add_command(label="Wordcloud", command=self.world_cloud)

        self.menubar.add_command(label="About", command=self.print_about)
        self.menubar.add_command(label="Exit", command=root.destroy)

        self.root.config(menu=self.menubar)

    def new_window(self, _class):
        self.new = Toplevel(self.root)
        _class(self.new)

    def scrap(self):
        self.new_window(Win0)

    def create_model(self):
        self.new_window(Win1)

    def neural_network(self):
        self.new_window(Win2)

    def naive_bayes(self):
        self.new_window(Win3)

    def dictionary(self):
        self.new_window(Win4)

    def world_cloud(self):
        self.new_window(Win5)

    def print_about(self):
        tkinter.messagebox.showinfo("Sentiment Analyzer", "BrandSentimentAnalyzer")

    def quit_window(self):
        self.root.destroy()
        sys.exit()


# SCRAP WINDOW
class Win0:

    def __init__(self, root):
        self.root = root
        self.root.title("Scrap")
        self.add_menu()
        # self.root.geometry("300x300+200+200")

        self.root.protocol("WM_DELETE_WINDOW", self.delete_window)

        Label(self.root, text="Website Link:").grid(row=0)
        Label(self.root, text="Tag:").grid(row=1)
        Label(self.root, text="Class:").grid(row=2)

        self.site = Entry(self.root)
        self.tag = Entry(self.root)
        self.cls = Entry(self.root)

        self.site.grid(row=0, column=1)
        self.tag.grid(row=1, column=1)
        self.cls.grid(row=2, column=1)

        Button(self.root, text='Scrap', command=self.scrap, width=16).grid(row=4, column=1, sticky=W, pady=5)

        self.root.focus_set()
        self.root.grab_set()

    def add_menu(self):

        self.menubar = Menu(self.root, tearoff=False)

        self.menubar.add_command(label="Save", command=self.save_file, state='disabled')
        self.menubar.add_command(label="Exit", command=self.delete_window)

        self.root.config(menu=self.menubar)

    csv_selected = FALSE
    output = 0

    def scrap(self):
        site_entry = self.site.get()
        tag_entry = self.tag.get()
        cls_entry = self.cls.get()

        if self.test(site_entry) == FALSE:
            tkinter.messagebox.showinfo("Sentiment Analyzer", "Website Address is not valid !")
        else:
            if tag_entry == "":
                tag_entry = 'p'

            r = requests.get(site_entry)
            soup = BeautifulSoup(r.content, features="lxml")
            text = ""
            for link in soup.find_all(tag_entry, {'class': cls_entry}):
                text = text + link.text

            text = text.split(".")
            df = pd.DataFrame(text, columns=['SentimentText'])
            tp = TextPreProcessing()
            df = tp.pre_process(df, 2)
            df.to_csv("out.csv", sep=',', encoding='utf-8')

            tkinter.messagebox.showinfo("Sentiment Analyzer", "Text Scraped!")
            self.menubar.entryconfig(0, state='active')

    def test(self, site):
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if re.match(regex, site) is not None:
            return TRUE
        else:
            return FALSE

    def save_file(self):
        files = [('csv file', '*.csv')]
        file = asksaveasfilename(filetypes=files, defaultextension=files)
        if file != "":
            data = pd.read_csv('out.csv')
            data.drop(data.columns[[0]], axis=1, inplace=True)
            data.to_csv(os.path.join(file))
            os.remove("out.csv")
            tkinter.messagebox.showinfo("CSV File", "Saved as csv")
            self.root.destroy()

    def delete_window(self):
        try:
            os.remove("out.csv")
        except OSError as e:
            pass

        try:
            self.root.destroy()
        except:
            pass


# CREATE MODEL WINDOW
class Win1:

    def __init__(self, root):
        self.root = root
        self.root.title("Create Model")
        self.add_menu()
        self.root.geometry("300x300+200+200")

        self.root.protocol("WM_DELETE_WINDOW", self.delete_window)

        self.root.focus_set()
        self.root.grab_set()

    def add_menu(self):

        self.menubar = Menu(self.root, tearoff=False)

        select = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="File", menu=select)

        select.add_command(label="Select CSV", command=self.training_file)

        self.menubar.add_command(label="Analyze", command=self.analyze, state='disabled')

        image = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="Model", menu=image, state='disabled')
        image.add_command(label="Save", command=self.save_file)

        self.menubar.add_command(label="Exit", command=self.delete_window)

        self.root.config(menu=self.menubar)

    csv_selected = FALSE
    output = 0

    def training_file(self):
        self.root.csv_filename = filedialog.askopenfilename(filetypes=[("csv files", "*.csv")])
        data = pd.read_csv(self.root.csv_filename)
        if self.root.csv_filename != "":
            with open(self.root.csv_filename, newline='') as f:
                reader = csv.reader(f)
                row1 = next(reader)
                count = sum(1 for row in reader)
            if 'Sentiment' not in row1:
                tkinter.messagebox.showerror("Error in csv File", "One of the column names must be 'Sentiment'!")
            elif 'SentimentText' not in row1:
                tkinter.messagebox.showerror("Error in csv File", "One of the column names must be 'SentimentText'!")
            elif count < 1:
                tkinter.messagebox.showerror("Error in csv File", "No Entry!")
            else:
                if 'Sentiment' in row1:
                    unique = set(data['Sentiment'])
                    Win1.output = len(list(unique))
                if Win1.output < 2 or Win1.output > 3:
                    tkinter.messagebox.showerror("Error in csv File",
                                                 "Only two or three type of sentiment is accepted!")
                else:
                    tkinter.messagebox.showinfo("CSV File", str("Selected CSV : " + str(self.root.csv_filename)))
                    Win1.csv_selected = TRUE
                    self.menubar.entryconfig(1, state='active')

    def analyze(self):

        data = pd.read_csv(self.root.csv_filename)

        data = data.sample(frac=1).reset_index(drop=True)
        data = data[['Sentiment', 'SentimentText']]

        # TEXT PRE-PROCESSING
        tp = TextPreProcessing()
        data = tp.pre_process(data, 3)

        tokenizer = Tokenizer(num_words=5000, split=" ")
        tokenizer.fit_on_texts(data['SentimentText'].values)

        x = tokenizer.texts_to_sequences(data['SentimentText'].values)
        x = pad_sequences(x, maxlen=32)

        y = pd.get_dummies(data['Sentiment']).values

        be.clear_session()

        model = Sequential()

        model.add(Embedding(5000, 256, input_length=x.shape[1]))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))
        model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(Win1.output, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        model.fit(x, y, epochs=8, batch_size=32, verbose=2)

        name = "model"
        extension = ".h5"

        model.save(name + extension)

        be.clear_session()

        tkinter.messagebox.showinfo("Model File", "Model is created!")

        self.menubar.entryconfig(2, state='active')

    def save_file(self):
        files = [('h5 file', '*.h5')]
        file = asksaveasfilename(filetypes=files, defaultextension=files)
        if file != "":
            model = load_model("model.h5")
            model.save(file)
            os.remove(os.path.join("model.h5"))
            tkinter.messagebox.showinfo("Model File", "Saved as h5")
            self.root.destroy()

    def delete_window(self):
        try:
            os.remove("model.h5")
        except OSError as e:
            pass

        try:
            self.root.destroy()
        except:
            pass


# ANALYZE USING NEURAL NETWORK WINDOW
class Win2:

    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network")
        self.add_menu()
        self.root.geometry("300x300+200+200")

        self.root.focus_set()
        self.root.grab_set()

    def add_menu(self):

        self.menubar = Menu(self.root, tearoff=False)

        select = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="File", menu=select)

        select.add_command(label="Select H5", command=self.training_file)
        select.add_command(label="Select CSV", command=self.testing_file)

        self.menubar.add_command(label="Analyze", command=self.analyze, state='disabled')

        image = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="Image", menu=image, state='disabled')
        image.add_command(label="Save", command=self.save_file)

        self.menubar.add_command(label="Exit", command=self.root.destroy)

        self.root.config(menu=self.menubar)

    csv_selected = FALSE
    h5_selected = FALSE

    def training_file(self):
        self.root.h5_filename = filedialog.askopenfilename(filetypes=[("h5 files", "*.h5")])
        if self.root.h5_filename != "":
            tkinter.messagebox.showinfo("Model File", str("Selected Model : " + str(self.root.h5_filename)))
            Win2.h5_selected = TRUE
            if Win2.csv_selected == TRUE:
                self.menubar.entryconfig(1, state='active')

    def testing_file(self):
        self.root.csv_filename = filedialog.askopenfilename(filetypes=[("csv files", "*.csv")])
        if self.root.csv_filename != "":
            with open(self.root.csv_filename, newline='') as f:
                reader = csv.reader(f)
                row1 = next(reader)
                count = sum(1 for row in reader)
            if 'SentimentText' not in row1:
                tkinter.messagebox.showerror("Error in csv File", "One of the column names must be 'SentimentText'!")
            elif count < 1:
                tkinter.messagebox.showerror("Error in csv File", "No Entry!")
            else:
                tkinter.messagebox.showinfo("Model File", str("Selected CSV : " + str(self.root.csv_filename)))
                Win2.csv_selected = TRUE
                if Win2.h5_selected == TRUE:
                    self.menubar.entryconfig(1, state='active')

    def analyze(self):

        pos_count, neg_count, neu_count = 0, 0, 0

        data = pd.read_csv(self.root.csv_filename)  # OS

        data = data.sample(frac=1).reset_index(drop=True)
        data = data[['SentimentText']]

        # TEXT PRE-PROCESSING
        tp = TextPreProcessing()
        data = tp.pre_process(data, 2)

        tokenizer = Tokenizer(num_words=5000, split=" ")
        tokenizer.fit_on_texts(data['SentimentText'].values)

        x = tokenizer.texts_to_sequences(data['SentimentText'].values)
        x = pad_sequences(x, maxlen=32)

        be.clear_session()
        model = load_model(self.root.h5_filename)
        predictions = model.predict(x)
        be.clear_session()

        output_dense = len(predictions[0])

        if output_dense == 2:
            for i, prediction in enumerate(predictions):
                if np.argmax(prediction) == 0:
                    neg_count += 1
                else:
                    pos_count += 1

            exp_vals = [pos_count, neg_count]
            exp_labels = ["positive", "negative"]

        else:
            for i, prediction in enumerate(predictions):
                if np.argmax(prediction) == 2:
                    pos_count += 1
                elif np.argmax(prediction) == 1:
                    neu_count += 1
                else:
                    neg_count += 1

            exp_vals = [pos_count, neu_count, neg_count]
            exp_labels = ["positive", "neutral", "negative"]

        actualfigure = plt.figure(figsize=(8, 8))
        actualfigure.suptitle("Sentiment Graph", fontsize=22)

        pie = plt.pie(exp_vals, labels=exp_labels, shadow=True, autopct='%1.1f%%')
        plt.legend(pie[0], exp_labels)

        canvas = FigureCanvasTkAgg(actualfigure, master=self.root)
        canvas.get_tk_widget().pack()
        canvas.draw()

        self.menubar.entryconfig(2, state='active')

    def save_file(self):
        files = [('png file', '*.png')]
        file = asksaveasfilename(filetypes=files, defaultextension=files)
        if file != "":
            plt.savefig(file)  # saves the image to the input file name.
            plt.clf()
            tkinter.messagebox.showinfo("Model File", "Saved as Png")
            self.root.destroy()


# ANALYZE USING NAIVE BAYES WINDOW
class Win3:

    def __init__(self, root):
        self.root = root
        self.root.title("Naive Bayes")
        self.add_menu()
        self.root.geometry("300x300+200+200")

        self.root.focus_set()
        self.root.grab_set()

    def add_menu(self):

        self.menubar = Menu(self.root, tearoff=False)

        select = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="File", menu=select)

        select.add_command(label="Select Trainig CSV", command=self.training_file)
        select.add_command(label="Select CSV", command=self.testing_file)

        self.menubar.add_command(label="Analyze", command=self.analyze, state='disabled')

        image = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="Image", menu=image, state='disabled')
        image.add_command(label="Save", command=self.save_file)

        self.menubar.add_command(label="Exit", command=self.root.destroy)

        self.root.config(menu=self.menubar)

    training_csv_selected = FALSE
    csv_selected = FALSE

    def training_file(self):
        self.root.training_filename = filedialog.askopenfilename(filetypes=[("csv files", "*.csv")])
        if self.root.training_filename != "":
            with open(self.root.training_filename, newline='') as f:
                reader = csv.reader(f)
                row1 = next(reader)
                count = sum(1 for row in reader)
            if 'Sentiment' not in row1:
                tkinter.messagebox.showerror("Error in csv File", "One of the column names must be 'Sentiment'!")
            elif 'SentimentText' not in row1:
                tkinter.messagebox.showerror("Error in csv File", "One of the column names must be 'SentimentText'!")
            elif count < 1:
                tkinter.messagebox.showerror("Error in csv File", "No Entry!")
            else:
                tkinter.messagebox.showinfo("Training File", str("Selected File : " + str(self.root.training_filename)))
                Win3.training_csv_selected = TRUE
                if Win3.csv_selected == TRUE:
                    self.menubar.entryconfig(1, state='active')

    def testing_file(self):
        self.root.csv_filename = filedialog.askopenfilename(filetypes=[("csv files", "*.csv")])
        if self.root.csv_filename != "":
            with open(self.root.csv_filename, newline='') as f:
                reader = csv.reader(f)
                row1 = next(reader)
                count = sum(1 for row in reader)
            if 'SentimentText' not in row1:
                tkinter.messagebox.showerror("Error in csv File", "One of the column names must be 'SentimentText'!")
            elif count < 1:
                tkinter.messagebox.showerror("Error in csv File", "No Entry!")
            else:
                tkinter.messagebox.showinfo("CSV File", str("Selected CSV : " + str(self.root.csv_filename)))
                Win3.csv_selected = TRUE
                if Win3.training_csv_selected == TRUE:
                    self.menubar.entryconfig(1, state='active')

    def analyze(self):

        pos_count, neg_count = 0, 0

        df = pd.read_csv(self.root.csv_filename)
        df_one = pd.read_csv(self.root.training_filename)

        # TEXT PRE-PROCESSING
        tp = TextPreProcessing()
        df = tp.pre_process(df, 3)
        df_one = tp.pre_process(df_one, 3)

        stopwords = set(sw.words('english'))
        vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopwords)

        y = pd.DataFrame(columns=['Sentiment'])

        for i in range(0, len(df_one)):
            if df_one.at[i, 'Sentiment'] == 'positive':
                y.loc[i] = int(1)
            else:
                y.loc[i] = int(0)

        X = vectorizer.fit_transform(df_one.SentimentText)
        y = y.astype('int')
        clf = naive_bayes.MultinomialNB()
        clf.fit(X, y.values.ravel())

        for i in range(0, len(df)):

            sent_text = str(df.at[i, 'SentimentText'])

            movie_array = np.array([sent_text])
            movie_review_vector = vectorizer.transform(movie_array)

            if int(clf.predict(movie_review_vector)) == 0:
                neg_count += 1
            else:
                pos_count += 1

        exp_vals = [pos_count, neg_count]
        exp_labels = ["positive", "negative"]

        actualfigure = plt.figure(figsize=(8, 8))
        actualfigure.suptitle("Sentiment Graph", fontsize=22)

        pie = plt.pie(exp_vals, labels=exp_labels, shadow=True, autopct='%1.1f%%')
        plt.legend(pie[0], exp_labels)

        canvas = FigureCanvasTkAgg(actualfigure, master=self.root)
        canvas.get_tk_widget().pack()
        canvas.draw()

        self.menubar.entryconfig(2, state='active')

    def save_file(self):
        files = [('png file', '*.png')]
        file = asksaveasfilename(filetypes=files, defaultextension=files)
        if file != "":
            plt.savefig(file)  # saves the image to the input file name.
            plt.clf()
            tkinter.messagebox.showinfo("Model File", "Saved as Png")
            self.root.destroy()


# ANALYZE USING DICTIONARY WINDOW
class Win4:

    def __init__(self, root):
        self.root = root
        self.root.title("Dictionary")
        self.add_menu()
        self.root.geometry("300x300+200+200")

        self.root.focus_set()
        self.root.grab_set()

    def add_menu(self):

        self.menubar = Menu(self.root, tearoff=False)

        select = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="File", menu=select)

        select.add_command(label="Select CSV", command=self.testing_file)

        self.menubar.add_command(label="Analyze", command=self.analyze, state='disabled')

        image = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="Image", menu=image, state='disabled')
        image.add_command(label="Save", command=self.save_file)

        self.menubar.add_command(label="Exit", command=self.root.destroy)

        self.root.config(menu=self.menubar)

    csv_selected = FALSE

    def testing_file(self):
        self.root.csv_filename = filedialog.askopenfilename(filetypes=[("csv files", "*.csv")])
        if self.root.csv_filename != "":
            with open(self.root.csv_filename, newline='') as f:
                reader = csv.reader(f)
                row1 = next(reader)
                count = sum(1 for row in reader)
            if 'SentimentText' not in row1:
                tkinter.messagebox.showerror("Error in csv File", "One of the column names must be 'SentimentText'!")
            elif count < 1:
                tkinter.messagebox.showerror("Error in csv File", "No Entry!")
            else:
                tkinter.messagebox.showinfo("CSV File", str("Selected CSV : " + str(self.root.csv_filename)))
                Win3.csv_selected = TRUE
                self.menubar.entryconfig(1, state='active')

    def analyze(self):

        positive_count, negative_count, neutral_count = 0, 0, 0
        df = pd.read_csv(self.root.csv_filename)

        # TEXT PRE-PROCESSING
        tp = TextPreProcessing()
        df = tp.pre_process(df, 2)

        negative_df = pd.read_csv("negative.csv")
        positive_df = pd.read_csv("positive.csv")

        for i in range(0, len(df)):

            string = df.at[i, 'SentimentText'].split(" ")

            pos_count = 0
            neg_count = 0

            for j in range(0, len(string)):

                word = string[j]
                word_index = string.index(string[j])

                for k in range(0, len(positive_df)):
                    positive_word = positive_df.at[k, 'Word']
                    if word == positive_word:
                        if word_index >= 1:
                            if string[word_index - 1] not in ["not", "isn't", "aren't", "wasn't", "weren't", "don't",
                                                              "didn't"]:
                                pos_count += 1
                            else:
                                neg_count += 1
                        else:
                            pos_count += 1

                for l in range(0, len(negative_df)):
                    negative_word = negative_df.at[l, 'Word']
                    if word == negative_word:
                        if word_index >= 1:
                            if string[word_index - 1] not in ["not", "isn't", "aren't", "wasn't", "weren't", "don't",
                                                              "didn't"]:
                                neg_count += 1
                            else:
                                pos_count += 1
                        else:
                            neg_count += 1

            if pos_count > neg_count:
                positive_count += 1
            elif neg_count > pos_count:
                negative_count += 1
            else:
                neutral_count += 1

        exp_vals = [positive_count, negative_count, neutral_count]
        exp_labels = ["positive", "negative", "neutral"]

        actualfigure = plt.figure(figsize=(8, 8))
        actualfigure.suptitle("Sentiment Graph", fontsize=22)

        pie = plt.pie(exp_vals, labels=exp_labels, shadow=True, autopct='%1.1f%%')
        plt.legend(pie[0], exp_labels)

        canvas = FigureCanvasTkAgg(actualfigure, master=self.root)
        canvas.get_tk_widget().pack()
        canvas.draw()

        self.menubar.entryconfig(2, state='active')

    def save_file(self):
        files = [('png file', '*.png')]
        file = asksaveasfilename(filetypes=files, defaultextension=files)
        if file != "":
            plt.savefig(file)  # saves the image to the input file name.
            plt.clf()
            tkinter.messagebox.showinfo("Model File", "Saved as Png")
            self.root.destroy()


# ANALYZE USING WORDCLOUD WINDOW
class Win5:

    def __init__(self, root):
        self.root = root
        self.root.title("WorldCloud")
        self.add_menu()
        self.root.geometry("300x300+200+200")

        self.root.focus_set()
        self.root.grab_set()

    def add_menu(self):

        self.menubar = Menu(self.root, tearoff=False)

        select = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="File", menu=select)

        select.add_command(label="Select CSV", command=self.testing_file)

        self.menubar.add_command(label="Analyze", command=self.analyze, state='disabled')

        image = Menu(self.root, tearoff=False)
        self.menubar.add_cascade(label="Image", menu=image, state='disabled')
        image.add_command(label="Save", command=self.save_file)

        self.menubar.add_command(label="Exit", command=self.root.destroy)

        self.root.config(menu=self.menubar)

    csv_selected = FALSE

    def testing_file(self):
        self.root.csv_filename = filedialog.askopenfilename(filetypes=[("csv files", "*.csv")])
        if self.root.csv_filename != "":
            with open(self.root.csv_filename, newline='') as f:  # APPLY THIS TO FLASK, REMOVE PUNCS, NUMS
                reader = csv.reader(f)
                row1 = next(reader)
                count = sum(1 for row in reader)
            if 'SentimentText' not in row1:
                tkinter.messagebox.showerror("Error in csv File", "One of the column names must be 'SentimentText'!")
            elif count < 1:
                tkinter.messagebox.showerror("Error in csv File", "No Entry!")
            else:
                tkinter.messagebox.showinfo("CSV File", str("Selected CSV : " + str(self.root.csv_filename)))
                Win3.csv_selected = TRUE
                self.menubar.entryconfig(1, state='active')

    def analyze(self):

        positive_count, negative_count, neutral_count = 0, 0, 0
        df = pd.read_csv(self.root.csv_filename)

        # TEXT PRE-PROCESSING
        tp = TextPreProcessing()
        df = tp.pre_process(df, 2)

        text = ""

        for i in range(0, len(df)):
            text = text + " " + df.at[i, 'SentimentText']

        wordcloud = WordCloud(
            width=3000,
            height=2000,
            background_color='black',
            stopwords=STOPWORDS).generate(str(text))

        actualfigure = plt.figure(
            figsize=(4, 3),
            facecolor='k',
            edgecolor='k')

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)

        canvas = FigureCanvasTkAgg(actualfigure, master=self.root)
        canvas.get_tk_widget().pack()
        canvas.draw()

        self.menubar.entryconfig(2, state='active')

    def save_file(self):
        files = [('png file', '*.png')]
        file = asksaveasfilename(filetypes=files, defaultextension=files)
        if file != "":
            plt.savefig(file)  # saves the image to the input file name.
            plt.clf()
            tkinter.messagebox.showinfo("Model File", "Saved as Png")
            self.root.destroy()


# TEXT PREPROCESSING
class TextPreProcessing:
    def pre_process(self, df, num):
        if num == 2:
            frame_new = pd.DataFrame(columns=['SentimentText'])
        elif num == 3:
            frame_new = pd.DataFrame(columns=['Sentiment', 'SentimentText'])

        content_array = []

        for index, row in islice(df.iterrows(), 0, None):

            content = str(row['SentimentText'])
            content = re.sub(
                r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
                " ", content)
            content = re.sub(r"http\S+", "", content)
            content = re.sub(r"@\S+", "", content)
            content = re.sub(r"#\S+", "", content)
            content = re.sub(r'\w*\d\w*', '', content).strip()
            content = re.sub(r'[^\w\s]', '', content)
            content = re.sub(' +', ' ', content)
            content = content.lower()

            if len(content) > 0:
                content_array.append(content)

                for i in range(0, len(content_array)):
                    if num == 2:
                        frame_new.at[i] = content_array[i]
                    elif num == 3:
                        frame_new.at[i, 'SentimentText'] = content_array[i]
                        frame_new.at[i, 'Sentiment'] = df.at[i, 'Sentiment']

        return frame_new


# RUN APP
if __name__ == "__main__":
    root = Tk()
    app = Win(root)
    app.root.title("Sentiment Analyzer")
    root.mainloop()
