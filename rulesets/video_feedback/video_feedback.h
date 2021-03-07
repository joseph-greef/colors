#ifndef _VIDEO_FEEDBACK_H
#define _VIDEO_FEEDBACK_H

#include <vector>

#include "board.cuh"
#include "ruleset.h"
#include "transformations/transformations.h"


class VideoFeedback : public Ruleset {
    private:
        Board<Pixel<float>> *current_frame_;
        Board<Pixel<float>> *last_frame_;
        Board<Pixel<float>> *temp_frame_;
        Board<Pixel<float>> *cudev_current_frame_;
        Board<Pixel<float>> *cudev_last_frame_;
        Board<Pixel<float>> *cudev_temp_frame_;

        std::vector<Transformation*> transformations_;

        void start_cuda();
        void stop_cuda();

        void randomize_effects();
    public:
        VideoFeedback(int width, int height);
        ~VideoFeedback();

        BoardType::BoardType board_get_type();
        BoardType::BoardType board_set_type();
        void* get_board();
        std::string get_name();
        void get_pixels(uint32_t *pixels);
        std::string get_rule_string();
        void load_rule_string(std::string rules);
        void print_human_readable_rules();
        void set_board(void *new_board);
        void start();
        void stop();
        void tick();
};

#endif //_VIDEO_FEEDBACK_H
