#include "HOG.hpp"


int main(void){
    // 예측
    HOG hog;
    Ptr<cv::ml::SVM> svm = Algorithm::load<cv::ml::SVM>("trained_svm_model.xml");

    string path = "./TestData/test/detect";
    for (const auto& entry : fs::directory_iterator(path)) {
        
        double scaleFactor = 1.5; 
        double currentScale = 1.0;
        int minSizeWidth = hog.getWidth(); 
        int minSizeHeight = hog.getHeight();

        Mat testImage = imread((string)entry.path());

        if(testImage.empty()){
            cout<<"Can't Open file"<<endl;
            continue;
        }

        int height = testImage.rows;
        int width = testImage.cols;
        
        vector<Rect> detectedRects;
        HOG hog;
        Mat resizeImage;
        resizeImage = testImage.clone();
        resize(resizeImage, resizeImage, Size(), 1.5, 1.5);

        while(minSizeWidth <= resizeImage.cols && minSizeHeight<=resizeImage.rows){
            
            resize(resizeImage, resizeImage, Size(), 1.0 / scaleFactor, 1.0 / scaleFactor);
            height = resizeImage.rows;
            width = resizeImage.cols;
            currentScale *= scaleFactor;

            for (int h = 0; h < height-hog.getHeight(); h += 5){

                for (int w = 0; w < width-hog.getWidth(); w += 5) {

                    Rect range(w,h, hog.getWidth(), hog.getHeight());
                    Mat scr = resizeImage(range);
                    // imshow("matching", scr);
                    // waitKey(0);
                    vector<float> descriptors = hog.getFeature(scr);
                    cv::Mat testFeatures = cv::Mat(descriptors).clone().reshape(1, 1);
                    float response = svm->predict(testFeatures);
                    if (response == 1) {
                        // cout<<"detect"<<endl;
                        int originalX = static_cast<int>(w * currentScale);
                        int originalY = static_cast<int>(h * currentScale);
                        int originalW = static_cast<int>(hog.getWidth() * currentScale);
                        int originalH = static_cast<int>(hog.getHeight() * currentScale);

                        // rectangle(testImage, Rect(originalX, originalY, originalW, originalH), Scalar(0, 255, 0), 2);
                        detectedRects.push_back(Rect(originalX, originalY, originalW, originalH));
                    }
                }

            }

        }
        vector<int> weights;
        groupRectangles(detectedRects, weights, 2, 0.3);

        // 최종 검출 결과 표시
        for (const auto& rect : detectedRects) {
            rectangle(testImage, rect, Scalar(0, 255, 0), 2);
        }

        imshow("Detected",testImage);
        waitKey(0);
    }

    return 0;
}