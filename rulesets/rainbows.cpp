
#include <ctime>
#include <climits>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "input_manager.h"
#include "rainbows.h"

Rainbows::Rainbows(int width, int height, int color_speed)
    : alive_color_scheme_(1)
    , alive_offset_(0)
    , changing_background_(false)
    , color_counter_(0)
    , color_offset_(0)
    , color_speed_(color_speed)
    , dead_color_scheme_(0)
    , dead_offset_(0)
    , gif_(NULL)
    , gif_frames_(0)
    , saved_alive_color_scheme_(2)
    , saved_dead_color_scheme_(2)
    , height_(height)
    , width_(width)
{
}

Rainbows::~Rainbows() {
}

void Rainbows::age_to_pixels(int *age_board, uint32_t *pixels) {
    for(int j = 0; j < height_; j++) {
        for(int i = 0; i < width_; i++) {
            int offset = j * width_ + i;
            if(age_board[offset] > 0) {
                pixels[offset] =
                    colors[alive_color_scheme_]
                        [(age_board[offset] + alive_offset_ + color_offset_) &
                         255];
            }
            else if(changing_background_ || age_board[offset] < 0) {
                pixels[offset] =
                    colors[dead_color_scheme_]
                        [(-age_board[offset] + dead_offset_ + color_offset_) &
                         255];
            }
            else {
                pixels[offset] =
                    colors[dead_color_scheme_]
                        [(-age_board[offset] + dead_offset_) & 255];
            }
        }
    }

    if(gif_) {
        save_gif_frame(age_board);
    }

    color_counter_++;
    if(color_speed_ > 0 && color_counter_ >= color_speed_) {
        color_counter_ = 0;
        color_offset_--;
    }
    else if (color_speed_ < 0) {
        color_offset_ += (color_speed_);
    }
}

void Rainbows::randomize_colors(bool control, bool shift) {
    alive_offset_ = rand() % RAINBOW_LENGTH;                                    
    color_offset_ = 0;
    dead_offset_ = rand() % RAINBOW_LENGTH;
}

void Rainbows::reset_colors(bool control, bool shift) {
    alive_offset_ = 0;                                    
    color_offset_ = 0;
    dead_offset_ = 0;
}

void Rainbows::save_gif_frame(int *age_board) {
    for(int i = 0; i < width_ * height_; i++) {
        if(age_board[i] > 0) {
            gif_->frame[i] = (age_board[i] + alive_offset_ + color_offset_) &
                             255;
        }
        if(age_board[i] > 0) {
            gif_->frame[i] = (-age_board[i] + dead_offset_ + color_offset_) &
                             255;
        }
        else {
            gif_->frame[i] = (-age_board[i] + dead_offset_) & 255;
        }
    }
    ge_add_frame(gif_, 2);

    gif_frames_--;
    if(gif_frames_ == 0) {
        ge_close_gif(gif_);
        gif_ = NULL;
    }
}

void Rainbows::start() { 
    InputManager::add_bool_toggler(&changing_background_, SDLK_b, "(RnBw) Toggle Changing Background");

    ADD_FUNCTION_CALLER(&Rainbows::randomize_colors, SDLK_BACKQUOTE,
                        "(RnBw) Randomize colors");
    ADD_FUNCTION_CALLER(&Rainbows::toggle_gif, SDLK_BACKSLASH,
                        "(RnBw) Toggle gif recording");
    ADD_FUNCTION_CALLER(&Rainbows::toggle_colors, SDLK_c,
                        "(RnBw) Toggle colors");
    ADD_FUNCTION_CALLER(&Rainbows::reset_colors, SDLK_l,
                        "(RnBw) Reset colors");

    InputManager::add_int_changer(&dead_color_scheme_,  SDLK_m, 0, Rainbows::num_colors-1, "(RnBw) Dead Scheme");
    InputManager::add_int_changer(&alive_color_scheme_, SDLK_n, 0, Rainbows::num_colors-1, "(RnBw) Alive Scheme");
    InputManager::add_int_changer(&dead_offset_,  SDLK_COMMA, INT_MIN, INT_MAX, "(RnBw) Dead Offset");
    InputManager::add_int_changer(&alive_offset_, SDLK_PERIOD, INT_MIN, INT_MAX, "(RnBw) Alive Offset");
    InputManager::add_int_changer(&color_speed_, SDLK_SLASH, INT_MIN, INT_MAX, "(RnBw) Color Speed");
}

void Rainbows::stop() { 
    InputManager::remove_var_changer(SDLK_b);

    InputManager::remove_var_changer(SDLK_BACKQUOTE);
    InputManager::remove_var_changer(SDLK_BACKSLASH);
    InputManager::remove_var_changer(SDLK_c);
    InputManager::remove_var_changer(SDLK_l);

    InputManager::remove_var_changer(SDLK_m);
    InputManager::remove_var_changer(SDLK_n);
    InputManager::remove_var_changer(SDLK_COMMA);
    InputManager::remove_var_changer(SDLK_PERIOD);
    InputManager::remove_var_changer(SDLK_SLASH);
}

void Rainbows::toggle_colors(bool control, bool shift) {
    int tmp_alive = alive_color_scheme_;
    int tmp_dead = dead_color_scheme_;
    alive_color_scheme_ = saved_alive_color_scheme_;
    dead_color_scheme_ = saved_dead_color_scheme_;
    saved_alive_color_scheme_ = tmp_alive;
    saved_dead_color_scheme_ = tmp_dead;
}

void Rainbows::toggle_gif(bool control, bool shift) {
    if(gif_) {
        ge_close_gif(gif_);
        gif_ = NULL;
    }
    else {
        static uint8_t rainbow_no_alpha[GIF_COLOR_LEN * 3] = { 0 };
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S.gif");
        std::string str = oss.str();

        for(int i = 0; i < GIF_COLOR_LEN; i++) {
            int nai = 3 * i;
            uint32_t color = colors[alive_color_scheme_][i];
            uint8_t *components = (uint8_t*)&color;
            rainbow_no_alpha[nai] = components[2];
            rainbow_no_alpha[nai + 1] = components[1];
            rainbow_no_alpha[nai + 2] = components[0];
        }

        gif_ = ge_new_gif(str.c_str(), width_, height_, rainbow_no_alpha, 8, 0);
        if(control) {
            gif_frames_ = 256;
        }
        else {
            gif_frames_ = 0;
        }
    }
}

uint32_t Rainbows::colors[][RAINBOW_LENGTH] = {
    //special case 0 and 1 are all black and all white for b/w drawing
    {0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000},
    {0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF, 0xFFFFFF},
    {0x000000, 0x070000, 0x0F0000, 0x170000, 0x1F0000, 0x270000, 0x2F0000, 0x370000, 0x3F0000, 0x470000, 0x4F0000, 0x570000, 0x5F0000, 0x670000, 0x6F0000, 0x770000, 0x7F0000, 0x870000, 0x8F0000, 0x970000, 0x9F0000, 0xA70000, 0xAF0000, 0xB70000, 0xBF0000, 0xC70000, 0xCF0000, 0xD70000, 0xDF0000, 0xE70000, 0xEF0000, 0xF70000, 0xFF0000, 0xFF0007, 0xFF000F, 0xFF0017, 0xFF001F, 0xFF0027, 0xFF002F, 0xFF0037, 0xFF003F, 0xFF0047, 0xFF004F, 0xFF0057, 0xFF005F, 0xFF0067, 0xFF006F, 0xFF0077, 0xFF007F, 0xFF0087, 0xFF008F, 0xFF0097, 0xFF009F, 0xFF00A7, 0xFF00AF, 0xFF00B7, 0xFF00BF, 0xFF00C7, 0xFF00CF, 0xFF00D7, 0xFF00DF, 0xFF00E7, 0xFF00EF, 0xFF00F7, 0xFF00FF, 0xF700FF, 0xEF00FF, 0xE700FF, 0xDF00FF, 0xD700FF, 0xCF00FF, 0xC700FF, 0xBF00FF, 0xB700FF, 0xAF00FF, 0xA700FF, 0x9F00FF, 0x9700FF, 0x8F00FF, 0x8700FF, 0x7F00FF, 0x7700FF, 0x6F00FF, 0x6700FF, 0x5F00FF, 0x5700FF, 0x4F00FF, 0x4700FF, 0x3F00FF, 0x3700FF, 0x2F00FF, 0x2700FF, 0x1F00FF, 0x1700FF, 0x0F00FF, 0x0700FF, 0x0000FF, 0x0007FF, 0x000FFF, 0x0017FF, 0x001FFF, 0x0027FF, 0x002FFF, 0x0037FF, 0x003FFF, 0x0047FF, 0x004FFF, 0x0057FF, 0x005FFF, 0x0067FF, 0x006FFF, 0x0077FF, 0x007FFF, 0x0087FF, 0x008FFF, 0x0097FF, 0x009FFF, 0x00A7FF, 0x00AFFF, 0x00B7FF, 0x00BFFF, 0x00C7FF, 0x00CFFF, 0x00D7FF, 0x00DFFF, 0x00E7FF, 0x00EFFF, 0x00F7FF, 0x00FFFF, 0x00FFF7, 0x00FFEF, 0x00FFE7, 0x00FFDF, 0x00FFD7, 0x00FFCF, 0x00FFC7, 0x00FFBF, 0x00FFB7, 0x00FFAF, 0x00FFA7, 0x00FF9F, 0x00FF97, 0x00FF8F, 0x00FF87, 0x00FF7F, 0x00FF77, 0x00FF6F, 0x00FF67, 0x00FF5F, 0x00FF57, 0x00FF4F, 0x00FF47, 0x00FF3F, 0x00FF37, 0x00FF2F, 0x00FF27, 0x00FF1F, 0x00FF17, 0x00FF0F, 0x00FF07, 0x00FF00, 0x07FF00, 0x0FFF00, 0x17FF00, 0x1FFF00, 0x27FF00, 0x2FFF00, 0x37FF00, 0x3FFF00, 0x47FF00, 0x4FFF00, 0x57FF00, 0x5FFF00, 0x67FF00, 0x6FFF00, 0x77FF00, 0x7FFF00, 0x87FF00, 0x8FFF00, 0x97FF00, 0x9FFF00, 0xA7FF00, 0xAFFF00, 0xB7FF00, 0xBFFF00, 0xC7FF00, 0xCFFF00, 0xD7FF00, 0xDFFF00, 0xE7FF00, 0xEFFF00, 0xF7FF00, 0xFFFF00, 0xFFFB00, 0xFFF800, 0xFFF400, 0xFFF100, 0xFFED00, 0xFFEA00, 0xFFE700, 0xFFE300, 0xFFE000, 0xFFDC00, 0xFFD901, 0xFFD601, 0xFFD201, 0xFFCF01, 0xFFCB01, 0xFFC801, 0xFFC501, 0xFFC101, 0xFFBE01, 0xFFBA01, 0xFFB701, 0xFFB402, 0xFFB002, 0xFFAD02, 0xFFA902, 0xFFA602, 0xFFA302, 0xFF9F02, 0xFF9C02, 0xFF9802, 0xFF9502, 0xFF9203, 0xF68D02, 0xEE8802, 0xE68302, 0xDE7F02, 0xD57A02, 0xCD7502, 0xC57102, 0xBD6C02, 0xB46702, 0xAC6202, 0xA45E01, 0x9C5901, 0x945401, 0x8B5001, 0x834B01, 0x7B4601, 0x734101, 0x6A3D01, 0x623801, 0x5A3301, 0x522F00, 0x4A2A00, 0x412500, 0x392000, 0x311C00, 0x291700, 0x201200, 0x180E00, 0x100900, 0x080400, 0x000000},
    {0xFF0000, 0xFF0005, 0xFF000B, 0xFF0011, 0xFF0017, 0xFF001D, 0xFF0023, 0xFF0029, 0xFF002F, 0xFF0035, 0xFF003B, 0xFF0041, 0xFF0047, 0xFF004D, 0xFF0053, 0xFF0058, 0xFF005E, 0xFF0064, 0xFF006A, 0xFF0070, 0xFF0076, 0xFF007C, 0xFF0082, 0xFF0088, 0xFF008E, 0xFF0094, 0xFF009A, 0xFF00A0, 0xFF00A6, 0xFF00AB, 0xFF00B1, 0xFF00B7, 0xFF00BD, 0xFF00C3, 0xFF00C9, 0xFF00CF, 0xFF00D5, 0xFF00DB, 0xFF00E1, 0xFF00E7, 0xFF00ED, 0xFF00F3, 0xFF00F9, 0xFF00FF, 0xF900FF, 0xF300FF, 0xED00FF, 0xE700FF, 0xE100FF, 0xDB00FF, 0xD500FF, 0xCF00FF, 0xC900FF, 0xC300FF, 0xBD00FF, 0xB700FF, 0xB100FF, 0xAB00FF, 0xA600FF, 0xA000FF, 0x9A00FF, 0x9400FF, 0x8E00FF, 0x8800FF, 0x8200FF, 0x7C00FF, 0x7600FF, 0x7000FF, 0x6A00FF, 0x6400FF, 0x5E00FF, 0x5800FF, 0x5300FF, 0x4D00FF, 0x4700FF, 0x4100FF, 0x3B00FF, 0x3500FF, 0x2F00FF, 0x2900FF, 0x2300FF, 0x1D00FF, 0x1700FF, 0x1100FF, 0x0B00FF, 0x0500FF, 0x0000FF, 0x0005FF, 0x000BFF, 0x0011FF, 0x0017FF, 0x001DFF, 0x0023FF, 0x0029FF, 0x002FFF, 0x0035FF, 0x003BFF, 0x0041FF, 0x0047FF, 0x004DFF, 0x0053FF, 0x0058FF, 0x005EFF, 0x0064FF, 0x006AFF, 0x0070FF, 0x0076FF, 0x007CFF, 0x0082FF, 0x0088FF, 0x008EFF, 0x0094FF, 0x009AFF, 0x00A0FF, 0x00A6FF, 0x00ABFF, 0x00B1FF, 0x00B7FF, 0x00BDFF, 0x00C3FF, 0x00C9FF, 0x00CFFF, 0x00D5FF, 0x00DBFF, 0x00E1FF, 0x00E7FF, 0x00EDFF, 0x00F3FF, 0x00F9FF, 0x00FFFF, 0x00FFF8, 0x00FFF2, 0x00FFEC, 0x00FFE6, 0x00FFE0, 0x00FFDA, 0x00FFD4, 0x00FFCE, 0x00FFC8, 0x00FFC2, 0x00FFBC, 0x00FFB6, 0x00FFB0, 0x00FFAA, 0x00FFA3, 0x00FF9D, 0x00FF97, 0x00FF91, 0x00FF8B, 0x00FF85, 0x00FF7F, 0x00FF79, 0x00FF73, 0x00FF6D, 0x00FF67, 0x00FF61, 0x00FF5B, 0x00FF55, 0x00FF4E, 0x00FF48, 0x00FF42, 0x00FF3C, 0x00FF36, 0x00FF30, 0x00FF2A, 0x00FF24, 0x00FF1E, 0x00FF18, 0x00FF12, 0x00FF0C, 0x00FF06, 0x00FF00, 0x06FF00, 0x0CFF00, 0x12FF00, 0x18FF00, 0x1EFF00, 0x24FF00, 0x2AFF00, 0x30FF00, 0x36FF00, 0x3CFF00, 0x42FF00, 0x48FF00, 0x4EFF00, 0x55FF00, 0x5BFF00, 0x61FF00, 0x67FF00, 0x6DFF00, 0x73FF00, 0x79FF00, 0x7FFF00, 0x85FF00, 0x8BFF00, 0x91FF00, 0x97FF00, 0x9DFF00, 0xA3FF00, 0xAAFF00, 0xB0FF00, 0xB6FF00, 0xBCFF00, 0xC2FF00, 0xC8FF00, 0xCEFF00, 0xD4FF00, 0xDAFF00, 0xE0FF00, 0xE6FF00, 0xECFF00, 0xF2FF00, 0xF8FF00, 0xFFFF00, 0xFFF800, 0xFFF200, 0xFFEC00, 0xFFE600, 0xFFE000, 0xFFDA00, 0xFFD400, 0xFFCE00, 0xFFC800, 0xFFC200, 0xFFBC00, 0xFFB600, 0xFFB000, 0xFFAA00, 0xFFA300, 0xFF9D00, 0xFF9700, 0xFF9100, 0xFF8B00, 0xFF8500, 0xFF7F00, 0xFF7900, 0xFF7300, 0xFF6D00, 0xFF6700, 0xFF6100, 0xFF5B00, 0xFF5500, 0xFF4E00, 0xFF4800, 0xFF4200, 0xFF3C00, 0xFF3600, 0xFF3000, 0xFF2A00, 0xFF2400, 0xFF1E00, 0xFF1800, 0xFF1200, 0xFF0C00, 0xFF0600, 0xFF0000},
    {0x1E00B0, 0x2101AD, 0x2502AA, 0x2804A7, 0x2C05A5, 0x2F07A2, 0x33089F, 0x360A9C, 0x3A0B9A, 0x3D0C97, 0x410E94, 0x440F91, 0x48118F, 0x4B128C, 0x4F1489, 0x521586, 0x561784, 0x591881, 0x5D197E, 0x601B7B, 0x641C79, 0x671E76, 0x6B1F73, 0x6E2170, 0x72226E, 0x75236B, 0x792568, 0x7C2665, 0x802863, 0x832960, 0x872B5D, 0x8A2C5A, 0x8E2E58, 0x922F55, 0x953052, 0x99324F, 0x9C334D, 0xA0354A, 0xA33647, 0xA73844, 0xAA3942, 0xAE3A3F, 0xB13C3C, 0xB53D39, 0xB83F37, 0xBC4034, 0xBF4231, 0xC3432E, 0xC6452C, 0xCA4629, 0xCD4726, 0xD14923, 0xD44A21, 0xD84C1E, 0xDB4D1B, 0xDF4F18, 0xE25016, 0xE65113, 0xE95310, 0xED540D, 0xF0560B, 0xF45708, 0xF75905, 0xFB5A02, 0xFF5C00, 0xFF5D00, 0xFF5F00, 0xFF6100, 0xFF6300, 0xFF6400, 0xFF6600, 0xFF6800, 0xFF6A00, 0xFF6B00, 0xFF6D00, 0xFF6F00, 0xFF7100, 0xFF7200, 0xFF7400, 0xFF7600, 0xFF7800, 0xFF7A00, 0xFF7B00, 0xFF7D00, 0xFF7F00, 0xFF8100, 0xFF8200, 0xFF8400, 0xFF8600, 0xFF8800, 0xFF8900, 0xFF8B00, 0xFF8D00, 0xFF8F00, 0xFF9000, 0xFF9200, 0xFF9400, 0xFF9600, 0xFF9800, 0xFF9900, 0xFF9B00, 0xFF9D00, 0xFF9F00, 0xFFA000, 0xFFA200, 0xFFA400, 0xFFA600, 0xFFA700, 0xFFA900, 0xFFAB00, 0xFFAD00, 0xFFAE00, 0xFFB000, 0xFFB200, 0xFFB400, 0xFFB600, 0xFFB700, 0xFFB900, 0xFFBB00, 0xFFBD00, 0xFFBE00, 0xFFC000, 0xFFC200, 0xFFC400, 0xFFC500, 0xFFC700, 0xFFC900, 0xFFCB00, 0xFFCD00, 0xFBCD02, 0xF7CE05, 0xF3CF08, 0xEFCF0B, 0xEBD00E, 0xE7D111, 0xE3D213, 0xDFD216, 0xDBD319, 0xD7D41C, 0xD3D41F, 0xCFD522, 0xCBD624, 0xC7D727, 0xC3D72A, 0xBFD82D, 0xBBD930, 0xB7D933, 0xB3DA36, 0xAFDB38, 0xABDC3B, 0xA7DC3E, 0xA3DD41, 0x9FDE44, 0x9BDE47, 0x97DF49, 0x93E04C, 0x8FE14F, 0x8BE152, 0x87E255, 0x83E358, 0x7FE45B, 0x7BE45D, 0x77E560, 0x73E663, 0x6FE666, 0x6BE769, 0x67E86C, 0x63E96E, 0x5FE971, 0x5BEA74, 0x57EB77, 0x53EB7A, 0x4FEC7D, 0x4BED7F, 0x47EE82, 0x43EE85, 0x3FEF88, 0x3BF08B, 0x37F08E, 0x33F191, 0x2FF293, 0x2BF396, 0x27F399, 0x23F49C, 0x1FF59F, 0x1BF5A2, 0x17F6A4, 0x13F7A7, 0x0FF8AA, 0x0BF8AD, 0x07F9B0, 0x03FAB3, 0x00FBB6, 0x00F7B5, 0x00F3B5, 0x01EFB5, 0x01EBB5, 0x02E7B5, 0x02E3B5, 0x03DFB5, 0x03DBB5, 0x04D7B5, 0x04D3B5, 0x05CFB4, 0x05CBB4, 0x06C7B4, 0x06C3B4, 0x07BFB4, 0x07BBB4, 0x08B7B4, 0x08B3B4, 0x09AFB4, 0x09ABB4, 0x0AA7B4, 0x0AA3B3, 0x0A9FB3, 0x0B9BB3, 0x0B97B3, 0x0C93B3, 0x0C8FB3, 0x0D8BB3, 0x0D87B3, 0x0E83B3, 0x0E7FB3, 0x0F7BB2, 0x0F77B2, 0x1073B2, 0x106FB2, 0x116BB2, 0x1167B2, 0x1263B2, 0x125FB2, 0x135BB2, 0x1357B2, 0x1453B2, 0x144FB1, 0x144BB1, 0x1547B1, 0x1543B1, 0x163FB1, 0x163BB1, 0x1737B1, 0x1733B1, 0x182FB1, 0x182BB1, 0x1927B0, 0x1923B0, 0x1A1FB0, 0x1A1BB0, 0x1B17B0, 0x1B13B0, 0x1C0FB0, 0x1C0BB0, 0x1D07B0, 0x1D03B0, 0x1E00B0},
    {0x27F550, 0x28F251, 0x2AEF53, 0x2CEC54, 0x2EE956, 0x2FE757, 0x31E459, 0x33E15A, 0x35DE5C, 0x37DB5D, 0x38D95F, 0x3AD660, 0x3CD362, 0x3ED063, 0x3FCE65, 0x41CB66, 0x43C868, 0x45C56A, 0x47C26B, 0x48C06D, 0x4ABD6E, 0x4CBA70, 0x4EB771, 0x50B473, 0x51B274, 0x53AF76, 0x55AC77, 0x57A979, 0x58A77A, 0x5AA47C, 0x5CA17D, 0x5E9E7F, 0x609B80, 0x619982, 0x639684, 0x659385, 0x679087, 0x698D88, 0x6A8B8A, 0x6C888B, 0x6E858D, 0x70828E, 0x718090, 0x737D91, 0x757A93, 0x777794, 0x797496, 0x7A7297, 0x7C6F99, 0x7E6C9A, 0x80699C, 0x82679E, 0x7F699F, 0x7D6BA0, 0x7B6EA2, 0x7970A3, 0x7773A5, 0x7475A6, 0x7278A7, 0x707AA9, 0x6E7DAA, 0x6C7FAC, 0x6982AD, 0x6784AE, 0x6587B0, 0x6389B1, 0x618CB3, 0x5E8EB4, 0x5C91B6, 0x5A93B7, 0x5896B8, 0x5698BA, 0x539BBB, 0x519DBD, 0x4FA0BE, 0x4DA2BF, 0x4BA5C1, 0x48A7C2, 0x46AAC4, 0x44ACC5, 0x42AFC6, 0x40B1C8, 0x3DB4C9, 0x3BB6CB, 0x39B9CC, 0x37BBCE, 0x35BECF, 0x32C0D0, 0x30C3D2, 0x2EC5D3, 0x2CC8D5, 0x2ACAD6, 0x27CDD7, 0x25CFD9, 0x23D2DA, 0x21D4DC, 0x1FD7DD, 0x1CD9DE, 0x1ADCE0, 0x18DEE1, 0x16E1E3, 0x14E3E4, 0x12E6E6, 0x16E4E1, 0x1BE2DC, 0x1FE0D8, 0x24DED3, 0x29DCCF, 0x2DDACA, 0x32D9C6, 0x37D7C1, 0x3BD5BD, 0x40D3B8, 0x45D1B4, 0x49CFAF, 0x4ECEAB, 0x53CCA6, 0x57CAA2, 0x5CC89D, 0x61C699, 0x65C494, 0x6AC290, 0x6EC18B, 0x73BF87, 0x78BD82, 0x7CBB7E, 0x81B979, 0x86B775, 0x8AB670, 0x8FB46C, 0x94B267, 0x98B063, 0x9DAE5E, 0xA2AC5A, 0xA6AB55, 0xABA951, 0xB0A74C, 0xB4A548, 0xB9A343, 0xBDA13F, 0xC29F3A, 0xC79E36, 0xCB9C31, 0xD09A2D, 0xD59828, 0xD99624, 0xDE941F, 0xE3931B, 0xE79116, 0xEC8F12, 0xF18D0D, 0xF58B09, 0xFA8904, 0xFF8800, 0xFA8502, 0xF68205, 0xF28008, 0xED7D0B, 0xE97A0E, 0xE57810, 0xE17513, 0xDC7216, 0xD87019, 0xD46D1C, 0xD06A1E, 0xCB6821, 0xC76524, 0xC36227, 0xBF5F2A, 0xBA5D2C, 0xB65A2F, 0xB25732, 0xAE5535, 0xA95238, 0xA5503A, 0xA14D3D, 0x9D4A40, 0x984843, 0x944546, 0x904248, 0x8C404B, 0x873D4E, 0x833A51, 0x7F3854, 0x7B3556, 0x763259, 0x722F5C, 0x6E2D5F, 0x6A2A62, 0x652764, 0x612567, 0x5D226A, 0x59206D, 0x541D70, 0x501A72, 0x4C1875, 0x481578, 0x43127B, 0x3F107E, 0x3B0D80, 0x370A83, 0x320886, 0x2E0589, 0x2A028C, 0x26008F, 0x26048D, 0x26098C, 0x260E8B, 0x26138A, 0x261888, 0x261C87, 0x262186, 0x262685, 0x262B83, 0x263082, 0x263481, 0x263980, 0x263E7E, 0x26437D, 0x26487C, 0x264C7B, 0x26517A, 0x265678, 0x265B77, 0x266076, 0x266475, 0x266973, 0x266E72, 0x267371, 0x267870, 0x267C6E, 0x26816D, 0x26866C, 0x268B6B, 0x269069, 0x269468, 0x269967, 0x269E66, 0x26A365, 0x26A863, 0x26AC62, 0x26B161, 0x26B660, 0x26BB5E, 0x26C05D, 0x26C45C, 0x26C95B, 0x26CE59, 0x26D358, 0x26D857, 0x26DC56, 0x26E154, 0x26E653, 0x26EB52, 0x26F051, 0x27F550},
    {0x285C2F, 0x275F2E, 0x26632D, 0x25672C, 0x246B2B, 0x236E2A, 0x227229, 0x217628, 0x207A27, 0x1F7E26, 0x1E8125, 0x1D8525, 0x1C8924, 0x1B8D23, 0x1A9122, 0x1A9421, 0x199820, 0x189C1F, 0x17A01E, 0x16A41D, 0x15A71C, 0x14AB1B, 0x13AF1B, 0x12B31A, 0x11B619, 0x10BA18, 0x0FBE17, 0x0EC216, 0x0DC615, 0x0DC914, 0x0CCD13, 0x0BD112, 0x0AD511, 0x09D911, 0x08DC10, 0x07E00F, 0x06E40E, 0x05E80D, 0x04EC0C, 0x03EF0B, 0x02F30A, 0x01F709, 0x00FB08, 0x00FF08, 0x05FD09, 0x0AFB0A, 0x0FF90B, 0x14F80C, 0x19F60D, 0x1EF40E, 0x23F20F, 0x28F110, 0x2DEF11, 0x32ED12, 0x38EC13, 0x3DEA14, 0x42E815, 0x47E616, 0x4CE517, 0x51E318, 0x56E119, 0x5BE01A, 0x60DE1B, 0x65DC1C, 0x6ADA1D, 0x70D91F, 0x75D720, 0x7AD521, 0x7FD322, 0x84D223, 0x89D024, 0x8ECE25, 0x93CD26, 0x98CB27, 0x9DC928, 0xA2C729, 0xA8C62A, 0xADC42B, 0xB2C22C, 0xB7C12D, 0xBCBF2E, 0xC1BD2F, 0xC6BB30, 0xCBBA31, 0xD0B832, 0xD5B633, 0xDBB535, 0xDBB039, 0xDCAC3E, 0xDDA843, 0xDEA447, 0xDF9F4C, 0xE09B51, 0xE09755, 0xE1935A, 0xE28F5F, 0xE38A63, 0xE48668, 0xE5826D, 0xE57E72, 0xE67A76, 0xE7757B, 0xE87180, 0xE96D84, 0xEA6989, 0xEA658E, 0xEB6092, 0xEC5C97, 0xED589C, 0xEE54A1, 0xEF4FA5, 0xEF4BAA, 0xF047AF, 0xF143B3, 0xF23FB8, 0xF33ABD, 0xF436C1, 0xF432C6, 0xF52ECB, 0xF62AD0, 0xF725D4, 0xF821D9, 0xF91DDE, 0xF919E2, 0xFA15E7, 0xFB10EC, 0xFC0CF0, 0xFD08F5, 0xFE04FA, 0xFF00FF, 0xF900FB, 0xF300F8, 0xED00F5, 0xE800F2, 0xE200EF, 0xDC00EC, 0xD600E9, 0xD100E6, 0xCB00E3, 0xC500E0, 0xBF00DC, 0xBA00D9, 0xB400D6, 0xAE00D3, 0xA800D0, 0xA300CD, 0x9D00CA, 0x9700C7, 0x9100C4, 0x8C00C1, 0x8600BE, 0x8000BA, 0x7B00B7, 0x7500B4, 0x6F00B1, 0x6900AE, 0x6400AB, 0x5E00A8, 0x5800A5, 0x5200A2, 0x4D009F, 0x47009B, 0x410098, 0x3B0095, 0x360092, 0x30008F, 0x2A008C, 0x240089, 0x1F0086, 0x190083, 0x130080, 0x0E017D, 0x12067A, 0x170C77, 0x1C1274, 0x211871, 0x261E6E, 0x2B236B, 0x302968, 0x352F65, 0x3A3562, 0x3F3B5F, 0x44405C, 0x494659, 0x4E4C56, 0x535253, 0x575850, 0x5C5D4D, 0x61634A, 0x666947, 0x6B6F44, 0x707541, 0x757B3E, 0x7A803B, 0x7F8638, 0x848C35, 0x899232, 0x8E982F, 0x939D2C, 0x98A329, 0x9CA926, 0xA1AF23, 0xA6B520, 0xABBA1D, 0xB0C01A, 0xB5C617, 0xBACC14, 0xBFD211, 0xC4D70E, 0xC9DD0B, 0xCEE308, 0xD3E905, 0xD8EF02, 0xDDF500, 0xD8F101, 0xD4ED02, 0xD0EA03, 0xCBE604, 0xC7E205, 0xC3DF06, 0xBEDB07, 0xBAD708, 0xB6D40A, 0xB1D00B, 0xADCC0C, 0xA9C90D, 0xA4C50E, 0xA0C20F, 0x9CBE10, 0x98BA11, 0x93B713, 0x8FB314, 0x8BAF15, 0x86AC16, 0x82A817, 0x7EA418, 0x79A119, 0x759D1A, 0x71991B, 0x6C961D, 0x68921E, 0x648F1F, 0x608B20, 0x5B8721, 0x578422, 0x538023, 0x4E7C24, 0x4A7926, 0x467527, 0x417128, 0x3D6E29, 0x396A2A, 0x34662B, 0x30632C, 0x2C5F2D, 0x285C2F},
    {0x955BBD, 0x935DBB, 0x915FB9, 0x8F61B8, 0x8E64B6, 0x8C66B5, 0x8A68B3, 0x886BB1, 0x876DB0, 0x856FAE, 0x8372AD, 0x8174AB, 0x8076A9, 0x7E79A8, 0x7C7BA6, 0x7A7DA5, 0x7980A3, 0x7782A1, 0x7584A0, 0x73869E, 0x72899D, 0x708B9B, 0x6E8D99, 0x6C9098, 0x6B9296, 0x699495, 0x679793, 0x659991, 0x649B90, 0x629E8E, 0x60A08D, 0x5EA28B, 0x5DA58A, 0x5BA788, 0x59A986, 0x57AB85, 0x56AE83, 0x54B082, 0x52B280, 0x50B57E, 0x4FB77D, 0x4DB97B, 0x4BBC7A, 0x49BE78, 0x48C076, 0x46C375, 0x44C573, 0x42C772, 0x41CA70, 0x3FCC6E, 0x3DCE6D, 0x3BD06B, 0x3AD36A, 0x38D568, 0x36D766, 0x34DA65, 0x33DC63, 0x31DE62, 0x2FE160, 0x2DE35E, 0x2CE55D, 0x2AE85B, 0x28EA5A, 0x26EC58, 0x25EF57, 0x27EB55, 0x29E754, 0x2CE453, 0x2EE052, 0x31DD50, 0x33D94F, 0x36D54E, 0x38D24D, 0x3BCE4C, 0x3DCB4A, 0x40C749, 0x42C448, 0x45C047, 0x47BC46, 0x4AB944, 0x4CB543, 0x4FB242, 0x51AE41, 0x54AB40, 0x56A73E, 0x59A33D, 0x5BA03C, 0x5E9C3B, 0x60993A, 0x639538, 0x659137, 0x688E36, 0x6A8A35, 0x6D8734, 0x6F8332, 0x728031, 0x747C30, 0x76782F, 0x79752E, 0x7B712C, 0x7E6E2B, 0x806A2A, 0x836729, 0x856328, 0x885F26, 0x8A5C25, 0x8D5824, 0x8F5523, 0x925122, 0x944D20, 0x974A1F, 0x99461E, 0x9C431D, 0x9E3F1C, 0xA13C1A, 0xA33819, 0xA63418, 0xA83117, 0xAB2D16, 0xAD2A14, 0xB02613, 0xB22312, 0xB51F11, 0xB71B10, 0xBA180E, 0xBC140D, 0xBF110C, 0xC10D0B, 0xC40A0A, 0xC1090C, 0xBE090E, 0xBB0910, 0xB80912, 0xB50914, 0xB20916, 0xAF0818, 0xAC081A, 0xA9081C, 0xA6081E, 0xA30820, 0xA00822, 0x9D0724, 0x9A0726, 0x970728, 0x94072A, 0x91072C, 0x8E072E, 0x8B0730, 0x880632, 0x850634, 0x830636, 0x800638, 0x7D063A, 0x7A063C, 0x77053E, 0x740540, 0x710542, 0x6E0544, 0x6B0546, 0x680548, 0x65054B, 0x62044D, 0x5F044F, 0x5C0451, 0x590453, 0x560455, 0x530457, 0x500359, 0x4D035B, 0x4A035D, 0x47035F, 0x450361, 0x420363, 0x3F0265, 0x3C0267, 0x390269, 0x36026B, 0x33026D, 0x30026F, 0x2D0271, 0x2A0173, 0x270175, 0x240177, 0x210179, 0x1E017B, 0x1B017D, 0x18007F, 0x150081, 0x120083, 0x0F0085, 0x0C0087, 0x090089, 0x07008C, 0x09018C, 0x0B028D, 0x0D048E, 0x10058F, 0x12078F, 0x140890, 0x160A91, 0x190B92, 0x1B0D93, 0x1D0E93, 0x1F0F94, 0x221195, 0x241296, 0x261496, 0x281597, 0x2B1798, 0x2D1899, 0x2F1A9A, 0x311B9A, 0x341C9B, 0x361E9C, 0x381F9D, 0x3A219D, 0x3D229E, 0x3F249F, 0x4125A0, 0x4327A1, 0x4628A1, 0x4829A2, 0x4A2BA3, 0x4C2CA4, 0x4F2EA4, 0x512FA5, 0x5331A6, 0x5532A7, 0x5834A8, 0x5A35A8, 0x5C36A9, 0x5E38AA, 0x6139AB, 0x633BAB, 0x653CAC, 0x673EAD, 0x6A3FAE, 0x6C41AF, 0x6E42AF, 0x7043B0, 0x7345B1, 0x7546B2, 0x7748B2, 0x7949B3, 0x7C4BB4, 0x7E4CB5, 0x804EB6, 0x824FB6, 0x8550B7, 0x8752B8, 0x8953B9, 0x8B55B9, 0x8E56BA, 0x9058BB, 0x9259BC, 0x955BBD},
};

int Rainbows::num_colors = sizeof(colors) / sizeof(colors[0]);

