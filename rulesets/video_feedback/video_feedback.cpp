
#include <iostream>

#include "cuda_runtime.h"

#include "input_manager.h"
#include "video_feedback.h"


VideoFeedback::VideoFeedback(int width, int height)
    : Ruleset(width, height)
{
    current_frame_ = new Board<Pixel<float>>(width, height, false);
    last_frame_ = new Board<Pixel<float>>(width, height, false);
    temp_frame_ = new Board<Pixel<float>>(width, height, false);
    cudev_current_frame_ = new Board<Pixel<float>>(width, height, true);
    cudev_last_frame_ = new Board<Pixel<float>>(width, height, true);
    cudev_temp_frame_ = new Board<Pixel<float>>(width, height, true);
}

VideoFeedback::~VideoFeedback() {
    for(Transformation *t : transformations_) {
        delete t;
    }

    delete current_frame_;
    delete last_frame_;
    delete temp_frame_;
    delete cudev_current_frame_;
    delete cudev_last_frame_;
    delete cudev_temp_frame_;
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
    Pixel<uint8_t> *pixel_board = (Pixel<uint8_t>*)pixels;

    if(use_gpu_) {
        current_frame_->copy_board_from(*cudev_current_frame_);
    }

    for(int i = 0; i < width_ * height_; i++) {
        pixel_board[i].part.r = current_frame_->data_[i].part.r;
        pixel_board[i].part.g = current_frame_->data_[i].part.g;
        pixel_board[i].part.b = current_frame_->data_[i].part.b;
        pixel_board[i].part.a = current_frame_->data_[i].part.a;
    }
    //TODO: copy Pixel<float>s into raw uint32_t pixels
    /*
    if(use_gpu_) {
        cudaMemcpy(pixels, cudev_current_frame_,
                   width_ * height_ * sizeof(cudev_current_frame_[0]),
                   cudaMemcpyDeviceToHost);
    }
    else {
        memcpy(pixels, current_frame_, width_ * height_ * sizeof(current_frame_[0]));
    }
    */
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

        /*
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
        */
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
        int r = rand() % 1;
        switch(r) {
            /*
            case 0:
                transformations_.push_back(new Blend(width_, height_));
                break;
            case 1:
                transformations_.push_back(new Brightness(width_, height_));
                break;
            case 2:
                transformations_.push_back(new Noise(width_, height_));
                break;
            case 3:
                transformations_.push_back(new Rotation(width_, height_));
                break;
            */
            case 0:
                transformations_.push_back(new Zoom(width_, height_));
                break;
        }
    }
}

//Takes an RGB board
void VideoFeedback::set_board(void* new_board) {
    Pixel<uint8_t> *pixel_board = (Pixel<uint8_t>*)new_board;
    for(int i = 0; i < width_ * height_; i++) {
        current_frame_->data_[i].part.r = pixel_board[i].part.r;
        current_frame_->data_[i].part.g = pixel_board[i].part.g;
        current_frame_->data_[i].part.b = pixel_board[i].part.b;
        current_frame_->data_[i].part.a = pixel_board[i].part.a;
    }
    cudev_current_frame_->copy_board_from(*current_frame_);

    /*
    memcpy(current_frame_, new_board, width_ * height_ * sizeof(current_frame_[0]));
    cudaMemcpy(cudev_current_frame_, new_board,
               width_ * height_ * sizeof(cudev_current_frame_[0]),
               cudaMemcpyHostToDevice);
    */
}


void VideoFeedback::start_cuda() {
    cudev_current_frame_->copy_board_from(*current_frame_);
    /*
    cudaMemcpy(cudev_current_frame_, current_frame_, width_ * height_ * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    */
}

void VideoFeedback::stop_cuda() {
    current_frame_->copy_board_from(*cudev_current_frame_);
    /*
    cudaMemcpy(current_frame_, cudev_current_frame_,
               width_ * height_ * sizeof(cudev_current_frame_[0]),
               cudaMemcpyDeviceToHost);
    */
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
    Board<Pixel<float>> **current, **last, **dest;
    if(use_gpu_) {
        cudev_last_frame_->copy_board_from(*cudev_current_frame_);
        /*
        cudaMemcpy(cudev_last_frame_, cudev_current_frame_,
                   width_ * height_ * sizeof(cudev_last_frame_[0]),
                   cudaMemcpyDeviceToDevice);
        */
        current = &cudev_current_frame_;
        last = &cudev_last_frame_;
        dest = &cudev_temp_frame_;
    }
    else {
        last_frame_->copy_board_from(*current_frame_);
        /*
        memcpy(last_frame_, current_frame_, width_ * height_ * sizeof(last_frame_[0]));
        */
        current = &current_frame_;
        last = &last_frame_;
        dest = &temp_frame_;
    }


    for(Transformation *t : transformations_) {
        Board<Pixel<float>> *tmp;
        std::cout << (*last)->data_ << std::endl;
        t->apply_transformation(**last, **current, **dest, use_gpu_);

        tmp = *current;
        *current = *dest;
        *dest = tmp;
    }
}


