
# DeOldify

#### Run deoldify through command-line without jupyter notebook or colab

##### Arguments:
````
--input           Input directory or file(image or video)
--output          Output directory
--factor          Render factor (default 21)
--model           Model weight type(artistic, stable, video)
--no_watermark    Disable watermark
--no_postprocess  Disable post-process
````
##### Run command:
```(bash)
python run_local.py --input ./test_images/1.mp4 --output ./test_images --model video --factor 21
```
