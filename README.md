# Description
## Restaurant Business Employee Tracking

This project is designed for a restaurant business use case. It tackles the challenge of monitoring café/restaurant staff and helps prevent any dishonest behavior.

## How It Works
- Employees wear a special uniform with shoulder patches in specific colors—each color corresponds to a particular employee.  
- The system tracks employees by these patches to monitor their movement and record attendance times.  
- It can also be used to detect cases where an employee has been absent for an extended period.  

## Additional Features
• Beyond color-coded patches, the module is capable of detecting employees by their facial features.  

Feel free to explore, contribute, and adapt this project to suit your restaurant's needs!

# Set Up
## Installation
```
pip install -r requirements.txt
```

## Load weights
Load weights for colour-patch model [here](https://drive.google.com/drive/folders/1iO7b_-0qvUCBF1SJEF5WAxAShXhPVyvj?usp=sharing) and put them in `./ckpts` folder.

## Set URLs of your cameras
In `main.py` you should set your URLs in **rtsp_urls** variable:
```
rtsp_urls = [
        'rtsp://<your url 1>',
        'rtsp://<your url 2>',
        'rtsp://<your url 3>',
        'rtsp://<your url 4>',
        ...
    ]
```

## Load faces of your staff
In `./staff` folder you can change names on another and also you should put in each name's folder at least 1 picture of corresponding person.

After that you should change next variables corresponding your staff's names:
- **names_labels** in `utils.py`
- **self.last_times** in `stream.py`
- **names_coords** in `processor.py`

# Inference
```
python main.py
```

# Example of work
<img src="example/example.gif" width="600">