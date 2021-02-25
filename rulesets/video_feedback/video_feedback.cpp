
#include <iostream>

#include "cuda_runtime.h"

#include "input_manager.h"
#include "video_feedback.h"


VideoFeedback::VideoFeedback(int width, int height)
    : Ruleset(width, height)
{
    current_frame_ = new Pixel[width_ * height_];
    last_frame_ = new Pixel[width_ * height_];
    temp_frame_ = new Pixel[width_ * height_];

    std::cout << "Allocating CUDA memory for VideoFeedback" << std::endl;

    cudaMalloc((void**)&cudev_current_frame_, width_ * height_ * sizeof(Pixel));
    cudaMalloc((void**)&cudev_dest_frame_, width_ * height_ * sizeof(Pixel));
    cudaMalloc((void**)&cudev_last_frame_, width_ * height_ * sizeof(Pixel));
}

VideoFeedback::~VideoFeedback() {
    delete[] current_frame_;
    delete[] last_frame_;
    delete[] temp_frame_;

    for(Transformation *t : transformations_) {
        delete t;
    }

    cudaFree((void*)cudev_current_frame_);
    cudaFree((void*)cudev_dest_frame_);
    cudaFree((void*)cudev_last_frame_);

    std::cout << "Freeing CUDA memory for VideoFeedback" << std::endl;
}

BoardType::BoardType VideoFeedback::board_get_type() {
    return BoardType::Other;
}

BoardType::BoardType VideoFeedback::board_set_type() {
    return BoardType::PixelBoard;
}

void* VideoFeedback::get_board() {
    return NULL;
}

std::string VideoFeedback::get_name() {
    return "VideoFeedback";
}
void VideoFeedback::get_pixels(uint32_t *pixels) {
    if(use_gpu_) {
        cudaMemcpy(pixels, cudev_current_frame_,
                   width_ * height_ * sizeof(cudev_current_frame_[0]),
                   cudaMemcpyDeviceToHost);
    }
    else {
        memcpy(pixels, current_frame_, width_ * height_ * sizeof(current_frame_[0]));
    }

}

std::string VideoFeedback::get_rule_string() {
    std::string rule_string = "";
    for(Transformation *t : transformations_) {
        rule_string += t->get_rule_string();
        rule_string += ";";
    }
    return rule_string;
}

void VideoFeedback::load_rule_string(std::string rules) {
    std::string rules_copy(rules);
    for(Transformation *t : transformations_) {
        delete t;
    }
    transformations_.clear();

    while(rules_copy.length() > 0) {
        size_t next_delim = rules_copy.find(';');
        std::string rule_string = rules_copy.substr(0, next_delim);
        std::string ruleset = rule_string.substr(0, rule_string.find(':'));
        std::string params = rule_string.substr(rule_string.find(':')+1);

        if(ruleset == "blend") {
            transformations_.push_back(new Blend(width_, height_, params));
        }
        if(ruleset == "brightness") {
            transformations_.push_back(new Brightness(width_, height_, params));
        }
        if(ruleset == "noise") {
            transformations_.push_back(new Noise(width_, height_, params));
        }
        if(ruleset == "rotation") {
            transformations_.push_back(new Rotation(width_, height_, params));
        }
        if(ruleset == "zoom") {
            transformations_.push_back(new Zoom(width_, height_, params));
        }

        rules_copy.erase(0, next_delim + 1);
    }
}

void VideoFeedback::print_human_readable_rules() {
    std::cout << get_rule_string() << std::endl;
}

void VideoFeedback::randomize_effects() {
    int num_effects = rand() % 4 + 2;
    for(Transformation *t : transformations_) {
        delete t;
    }
    transformations_.clear();

    for(int i = 0; i < num_effects; i++) {
        int r = rand() % 4;
        switch(r) {
            case 0:
                transformations_.push_back(new Rotation(width_, height_));
                break;
            case 1:
                transformations_.push_back(new Zoom(width_, height_));
                break;
            case 2:
                transformations_.push_back(new Noise(width_, height_));
                break;
            case 3:
                transformations_.push_back(new Blend(width_, height_));
                break;
            case 4:
                transformations_.push_back(new Brightness(width_, height_));
                break;
        }
    }
}

void VideoFeedback::set_board(void* new_board) {
    memcpy(current_frame_, new_board, width_ * height_ * sizeof(current_frame_[0]));
    cudaMemcpy(cudev_current_frame_, new_board,
               width_ * height_ * sizeof(cudev_current_frame_[0]),
               cudaMemcpyHostToDevice);
}


void VideoFeedback::start_cuda() {
    cudaMemcpy(cudev_current_frame_, current_frame_, width_ * height_ * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void VideoFeedback::stop_cuda() {
    cudaMemcpy(current_frame_, cudev_current_frame_,
               width_ * height_ * sizeof(cudev_current_frame_[0]),
               cudaMemcpyDeviceToHost);
}


void VideoFeedback::start() {
    std::cout << "Starting VideoFeedback" << std::endl;
    Ruleset::start();

    ADD_FUNCTION_CALLER(&VideoFeedback::randomize_effects, SDL_SCANCODE_R, false, false,
                        "VideoFeedback", "Randomize effects");
}

void VideoFeedback::stop() {
    Ruleset::stop();

    InputManager::remove_var_changer(SDL_SCANCODE_R, false, false);
}

void VideoFeedback::tick() {
    Pixel **current, **last, **dest;
    if(use_gpu_) {
        cudaMemcpy(cudev_last_frame_, cudev_current_frame_,
                   width_ * height_ * sizeof(cudev_last_frame_[0]),
                   cudaMemcpyDeviceToDevice);
        current = &cudev_current_frame_;
        last = &cudev_last_frame_;
        dest = &cudev_dest_frame_;
    }
    else {
        memcpy(last_frame_, current_frame_, width_ * height_ * sizeof(last_frame_[0]));
        current = &current_frame_;
        last = &last_frame_;
        dest = &temp_frame_;
    }


    for(Transformation *t : transformations_) {
        Pixel *tmp;
        t->apply_transformation(*last, *current, *dest, use_gpu_);

        tmp = *current;
        *current = *dest;
        *dest = tmp;
    }
}


