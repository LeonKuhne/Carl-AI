Carl is an AI that learns what monitor you're looking at using depth vision data.

### Required Hardware
- Kinect v1

### Setup Istructions
- Install libfreenect
- > mkdir models

### Run
- collect.py [-a <collection-name>] # collect user data
- carl.py # train model on collected data
- run.py # use trained model to predict viewed display, focus to display
- view.py # show a color interperetation of the depth video footage that is being collected 

### Testing
- LSTM implementation; Carl learns recurrently, taking in a sequence of frames per evaluation.

### Future Considerations
- Camera Pixel Data; In conjunction with depth data, use pixel color data as well.
