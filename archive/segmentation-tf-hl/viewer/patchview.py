#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import sys
import os
#from mpi4py import MPI
#import adios2 as ad
#from plxr import *

viewheight = 400
viewwidth = 400

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def tree_item_clicked(self, event):
        image_name = self.bptree.item(self.bptree.focus())['text']
        print ("Clicked it " + image_name)

        image_path = "%s/image/%s"%(sys.argv[1],image_name)
        print (image_path)

        mask_path = "%s/mask/%s"%(sys.argv[1],image_name)
        print (mask_path)

        pilImage = Image.open(image_path)
        pilImage = pilImage.resize ((viewwidth*2, viewheight*2), Image.ANTIALIAS)
        self.cur_img = ImageTk.PhotoImage(pilImage)
        self.canvas.create_image(0,0,image=self.cur_img)

        pilImage = Image.open(mask_path)
        pilImage = pilImage.resize ((viewwidth*2, viewheight*2), Image.ANTIALIAS)
        self.mask_img = ImageTk.PhotoImage(pilImage)
        self.maskcanvas.create_image(0,0,image=self.mask_img)
        self.master.update()

    def create_widgets(self):
        self.winfo_toplevel().title(sys.argv[1])

        self.bptree = ttk.Treeview(self)
        #self.bptree.pack(side="left")
        self.bptree.grid(row=0, column=0, sticky=tk.E+tk.W+tk.N+tk.S)
        self.bptree['columns'] = ('Type')
        self.bptree.bind("<ButtonRelease-1>", self.tree_item_clicked)

        self.canvas = tk.Canvas(self, width=viewwidth,height=viewheight)
        #self.canvas.pack(side='right')
        self.canvas.grid(row=0, column=1, sticky=tk.E+tk.W+tk.N+tk.S)

        self.maskcanvas = tk.Canvas(self, width=viewwidth,height=viewheight)
        self.maskcanvas.grid(row=0, column=2, sticky=tk.E+tk.W+tk.N+tk.S)

        for name in os.listdir("%s/image"%sys.argv[1]):
            self.bptree.insert('', 'end', name, text=name)

        self.quit = tk.Button(self, text="Exit", fg="red",
                              command=self.master.destroy)
        #self.quit.pack(side="bottom")
        self.quit.grid(row=1, column=0)


def usage():
    print("Add usage text...")


def main():

    #Check for single argument
    if len(sys.argv) < 2:
        usage()
        exit(1)

    #print (sys.argv[1])
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

def usage():
    print ("Usage: patchview.py <directory>")
    print ("    <directory> should contain 'image' and 'mask' subdirectories")


if __name__ == "__main__":

    if (len(sys.argv)) < 2:
        usage()
        exit(1)


    main()


