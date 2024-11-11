#include "HOG.hpp"

int main(void){
    // 예측
    HOG hog;
    Ptr<cv::ml::SVM> svm = Algorithm::load<cv::ml::SVM>("trained_svm_model.xml");

    string pathImage = "./Test/Test/JPEGImages"; 

    int cnt=0;
    int correct=0;
    try {
        for (const auto& entry : fs::directory_iterator(pathImage)) {
            if (entry.is_regular_file()) { // 일반 파일인 경우만
                // Mat testImage = imread(entry.path());

                // std::vector<float> descriptors = hog.getFeature(testImage); 

                // cv::Mat testFeatures = cv::Mat(descriptors).clone().reshape(1, 1);
                // float response = svm->predict(testFeatures);
                // if (response == 1) {
                //     std::cout << "Positive class" << std::endl;
                //     correct++;
                // } else {
                //     std::cout << "Negative class" << std::endl;
                // }
                // cnt++;
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }



    




}