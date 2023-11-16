function displayResizedImage(img, resizedImg){
    cv.resize(img, resizedImg, new cv.Size(740, 540), 0,0, cv.INTER_LINEAR);
    cv.imshow("main-canvas", resizedImg);
};

function opencvReady(){
    cv["onRuntimeInitialized"] = () =>
    {
        console.log("OpenCV Ready");
        const labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light","fire hydrant", "stop sign", "parking meter", "bench", "bird","cat", "dog", "horse", "sheep", "cow","elephant", "bear", "zebra", "giraffe", "backpack","umbrella", "handbag", "tie", "suitcase", "frisbee","skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle","wine glass", "cup", "fork", "knife", "spoon","bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot", "hot dog", "pizza", "donut","cake", "chair", "couch", "potted plant", "bed","dining table", "toilet", "tv", "laptop", "mouse","remote", "keyboard", "cell phone", "microwave", "oven","toaster", "sink", "refrigerator", "book", "clock","vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        //Read an Image from the image source and covert it into the OpenCV Format
        //In JS we use Let to define our variable
        let imgMain = cv.imread("img-main");
        cv.imshow("main-canvas", imgMain);
        imgMain.delete();

        //RGB Image
        document.getElementById("RGB-Image").onclick = function(){
            console.log("RGB Image");
            let imgMain = cv.imread("img-main");
            let imgGray = new cv.Mat();
            cv.cvtColor(imgMain, imgGray,cv.COLOR_RGBA2GRAY);
            cv.imshow("main-canvas", imgMain);
            imgMain.delete();
            imgGray.delete();
        }
        //Gray Scale Image
        document.getElementById("Gray-Image").onclick = function(){
            console.log("Gray Scale Image");
            let imgMain = cv.imread("img-main");
            let imgGray = new cv.Mat();
            cv.cvtColor(imgMain, imgGray,cv.COLOR_RGBA2GRAY);
            cv.imshow("main-canvas", imgGray);
            imgMain.delete();
            imgGray.delete();
        }

        //Objects Detection in an Image
        document.getElementById("Objects-Detection-Image").onclick = async function(){
            console.log("Objects Detection on Image");
            const numClass = 80;
            let image = document.getElementById('img-main');
            let inputimg = cv.imread("input-image");
            const resizedImg = new cv.Mat();
            // Load the TensorFlow.js graph model
            let model = await tf.loadGraphModel('yolov8n_web_model/model.json');
            console.log("Model Loaded", model);
            const modelWidth = 640;
            const modelHeight = 640;
            const preprocess = (image, modelWidth, modelHeight) => {
              let xRatio, yRatio;
              const input = tf.tidy(() => {
                // Convert pixel data from an image source (like an HTML image element or a canvas) into a TensorFlow.js tensor
                const  img = tf.browser.fromPixels(image);
                //Extracting the height (h) and width (w) of the image tensor (img). 
                const [h,w] = img.shape.slice(0,2);
                console.log("h & w", h, w);
                //calculates the maximum value between w and h
                const maxSize = Math.max(w, h); 
                const imgPadded = img.pad([
                  [0, maxSize - h], // padding y [bottom only]
                  [0, maxSize - w], // padding x [right only]
                  [0, 0],
                ]);
            
                xRatio = maxSize / w; // update xRatio
                yRatio = maxSize / h; // update yRatio
                return tf.image
                .resizeBilinear(imgPadded, [modelWidth, modelHeight]) 
                .div(255.0) 
                .expandDims(0); 
              });
            return [input, xRatio, yRatio];
            };
            const [input, xRatio, yRatio] = preprocess(image, modelWidth, modelHeight); 
            console.log("X-ratio, Y-ratio", xRatio, yRatio)
            const res = model.predict(input); // inference model
            const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
            const boxes = tf.tidy(() => {
              const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
              const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
              const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
              const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
              return tf
                .concat(
                  [
                    y1,
                    x1,
                    tf.add(y1, h), //y2
                    tf.add(x1, w), //x2
                  ],
                  2
                )
                .squeeze();
            }); 
            const [scores, classes] = tf.tidy(() => {
              const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); 
              return [rawScores.max(1), rawScores.argMax(1)];
            }); 
            const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes
            const predictionsLength = nms.size;
            console.log("Predictions Length", predictionsLength)
            if (predictionsLength > 0) {
                const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
                const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
                const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index
                console.log(boxes_data, scores_data, classes_data, [xRatio, yRatio]);
                for (let i = 0; i < scores_data.length; ++i) {
                    const klass = labels[classes_data[i]];
                    const color = [255,0,0,255];
                    const score = (scores_data[i] * 100).toFixed(1);
                    let [y1, x1, y2, x2] = boxes_data.slice(i * 4, (i + 1) * 4);
                    x1 *= xRatio;
                    x2 *= xRatio;
                    y1 *= yRatio;
                    y2 *= yRatio;
                    const width = x2 - x1;
                    const height = y2 - y1;
                    console.log(x1, y1, width, height,klass, score);
                    let point1 = new cv.Point(x1, y1)
                    let point2 = new cv.Point(x1 + width, y1 + height);
                    cv.rectangle(inputimg, point1, point2, [0, 0,255, 255],4);
                    const text = `${klass} - ${Math.round(score)/100}`;
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    const textMetrics = context.measureText(text);
                    const twidth = textMetrics.width;
                    console.log("Text Width",twidth)
                    cv.rectangle(inputimg, new cv.Point(x1,y1-30), new cv.Point(x1+ twidth +150, y1), [0, 0,255, 255],-1)
                    cv.putText(inputimg, text, new cv.Point(x1, y1 - 5),cv.FONT_HERSHEY_TRIPLEX, 0.70, new cv.Scalar(255,255,255,255), 1);
                }
                displayResizedImage(inputimg, resizedImg);
            }
            else{
                displayResizedImage(inputimg, resizedImg);
            }
            tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory
            resizedImg.delete();
      
        };

        //Object Detection in an Video
        document.getElementById("Objects-Detection-Video").onclick = function(){
            console.log("Object Detection Video");
            const video = document.getElementById('webcam');
            const enableWebcamButton = document.getElementById('enableWebcambutton');
            let model = undefined;
            let streaming = false;
            let src;
            let cap;
            let resizedImg;       
            if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
              enableWebcamButton.addEventListener('click', ()=>{
                if(!streaming){
                  console.log("Streaming Started");
                  enableCam();
                  streaming=true;
                }
                else{
                  console.log("Streaming Paused")
                  video.pause();
                  video.srcObject.getTracks().forEach(track => track.stop());
                  video.srcObject=null;
                  streaming=false;
        
                }
              })
            } else {
              console.warn('getUserMedia() is not supported by your browser');
            }
            function enableCam() {
              if (!model) {
                return;
              }
              navigator.mediaDevices.getUserMedia({'video':true, 'audio':false}).then(function(stream) {
                video.srcObject = stream;
                video.addEventListener('loadeddata', predictWebcam);
              });
            }
        
            setTimeout(async function(){
              try{
                model = await tf.loadGraphModel("yolov8n_web_model/model.json");
              } catch(error){
                console.error("Error Loading YOLOv8 Tf.js model", error);
              }
            }, 0);
        
            async function predictWebcam() {
                // Check if the video element has loaded data
              if (!video || !video.videoWidth || !video.videoHeight) {
                return;
              }
              const numClasses = 80;
              const begin = Date.now();
              console.log("Video Height & Width", video.height, video.width)
              src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
              cap = new cv.VideoCapture(video);
              cap.read(src);
              resizedImg = new cv.Mat();
              const modelWidth = 640;
              const modelHeight = 640;
              const preprocess = (video, modelWidth, modelHeight) => {
                  let xRatio, yRatio;
                  const input = tf.tidy(() => {
                      const  img = tf.browser.fromPixels(video);
                      const [h,w] = img.shape.slice(0,2);
                      const maxSize = Math.max(w, h); 
                      const imgPadded = img.pad([
                          [0, maxSize - h], // padding y [bottom only]
                          [0, maxSize - w], // padding x [right only]
                          [0, 0],
                        ]);
                        xRatio = maxSize / w; // update xRatio
                        yRatio = maxSize / h; // update yRatio
                      return tf.image.resizeBilinear(imgPadded, [modelWidth, modelHeight]). div(255.0).expandDims(0);
                  });
              return [input, xRatio, yRatio];
              }
              const [input, XRatio, yRatio] = preprocess(video, modelWidth, modelHeight); 
              console.log("X-Ratio, y-Ratio", XRatio, yRatio)
              const res = model.predict(input);
              const transRes = res.transpose([0,2,1]);
              const boxes = tf.tidy(() => {
                  const w = transRes.slice([0,0,2], [-1,-1,1]); //get width
                  const h = transRes.slice([0,0,3], [-1,-1,1]);//get height
                  const x1 = tf.sub(transRes.slice([0,0,0], [-1,-1,1]), tf.div(w,2));
                  const y1 = tf.sub(transRes.slice([0,0,1], [-1,-1,1]), tf.div(h,2));
                  return tf.concat([y1, x1, tf.add(y1,h), tf.add(x1, w)], 2).squeeze();//y1, x1, y2, x2
              });
              const [scores, classes] = tf.tidy(() => {
                  const rawscores = transRes.slice([0,0,4], [-1,-1, numClasses]).squeeze(0);
                  return [rawscores.max(1), rawscores.argMax(1)];
              });
              const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.5);
              const predictionsLength = nms.size;
              console.log("Predictions Length", predictionsLength)
              if (predictionsLength > 0) {
                const boxes_data = boxes.gather(nms, 0).dataSync();
                const scores_data = scores.gather(nms, 0).dataSync();
                const classes_data = classes.gather(nms, 0).dataSync();
                console.log(boxes_data, scores_data, classes_data);
        
                for (let i = 0; i < scores_data.length; i++) {
                  const className = labels[classes_data[i]];
                  const score = (scores_data[i] * 100).toFixed(1);
                  let [y1, x1, y2, x2] = boxes_data.slice(i*4, (i+1)*4);
                  x1 *= XRatio;
                  x2 *= XRatio;
                  y1 *= yRatio;
                  y2 *= yRatio;
                  const width = x2 - x1;
                  const height = y2 - y1;
                  console.log(x1, y1, width, height,className, score);
                  // Draw bounding box
                  let point1 = new cv.Point(x1, y1)
                  let point2 = new cv.Point(x1 + width, y1 + height);
                  cv.rectangle(src, point1, point2, [0, 0, 255, 255], 4);
      
                  // Draw text background
                  const text = `${className} - ${Math.round(score) / 100}`;
                  const canvas = document.createElement('canvas');
                  const context = canvas.getContext('2d');
                  const textMetrics = context.measureText(text);
                  const twidth = textMetrics.width;
                  cv.rectangle(src, new cv.Point(x1, y1 - 30), new cv.Point(x1 + twidth + 150, y1), [0, 0, 255, 255], -1);
                  cv.putText(src, text, new cv.Point(x1, y1 - 5), cv.FONT_HERSHEY_TRIPLEX, 0.70, new cv.Scalar(255, 255, 255, 255), 1);
                }
                displayResizedImage(src, resizedImg);
              } 
              else 
              {
                displayResizedImage(src, resizedImg);
              }
        
              // Clear memory
              tf.dispose([res, transRes, boxes, scores, classes, nms]);
        
              // Call the function again after a delay
              const delay = 1000 / 24 - (Date.now() - begin);
              setTimeout(predictWebcam, delay);
        
              // Release the source image
              src.delete();
              resizedImg.delete();
            }
        }


    }
}