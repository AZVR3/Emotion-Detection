# Emotion Detection Project

This is a small project that was completed as practice to using **tenserflow** and **opencv**.
I have really enjoyed this project overall and hope this helps out whoever is starting
out on learning opencv for their own smaller projects.

---

## Image training
The dataset was gathered from <a href="https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset">kaggle</a>. Completed by _Johnathan Oheix_.
<br><br>
The image training can be configured through the following steps:
1. Organize the image datas.
   + Categorize them into its own respectful emotions.
   + The image data that you personally take will work better for your particular face but may not be as accurate for others.
2. Rearrange your organized dataset into a new folder.
3. Reroute the directory in **<model.py>** to the desired training folder.
   + Or you can simply replace the dataset in [images/training] if desired to do so.
4. Rerun the **<model.py>** to update the model to the new dataset.
5. Voila!

---

※ NEVER CONFIGURE THE **<haarcascade_frontalface_default.xml>** AS THAT WOULD MAKE THE MODEL NOT BE ABLE TO UPDATE ITSELF.

---

## Closing notes

You are free to use this program as you want.
This program will not be accepting any form of branches but comments will 
be read when sent. 
<br><br>
Happy coding!