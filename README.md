# Fake news detector
Fake news detector in Python. Machine learning project using Flask API.

<p align="center">
      <img src="https://github.com/LukaszMikol/fake_news_detector/blob/master/images/detector.png" alt="Detector localhost" title="Detector localhost">
</p>
<br>

## Data from Kaggle
<p>Link: https://www.kaggle.com/c/fake-news/data</p>
 
## The repository contains:
<ol>
	<li><b><i>app.py</i></b> - main application</li>
	<li><b><i>fake_news_detector_prepare_model.ipynb</i></b> - Jupyter notebook file with models creation</li>
	<li><b><i>test_data</i></b> - folder with examples of false and real news</li>
    <li><b><i>requirements.txt</i></b> - required libraries to install</li>
	<li><b><i>Dockerfile</i></b></li>
</ol>

## Running the project using docker:
<ol>
	<li>
		<p>Build image</p>
		<b>docker build -t fake_news_detector_app:v1 .</b>
	</li>
	<li>
		<p>Running container</p>
		<b>docker container run -d -p 5000:5000 fake_news_detector_app:v1</b>
		<p>Navigate to URL http://localhost:5000</p>
	</li>
	<li>
		<p>Stop the container</p>
		<b>docker stop containter_name</b>
	</li>
</ol>


