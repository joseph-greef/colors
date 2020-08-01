
#include "initializer.h"
#include "rainbows.h"
#include "ruleset.h"


class LifeLike : public Ruleset {
    private:
        int alive_color_scheme_;
        int alive_offset_;
        int *board_;
        int *board_buffer_;
        bool born_[9];
        int dead_color_scheme_;
        int dead_offset_;
        bool draw_color_;
        Initializer initializer_;
        int num_faders_;
        Rainbows rainbows_;
        bool stay_alive_[9];

        void randomize_ruleset();
    public:
        LifeLike(int width, int height);
        ~LifeLike();
        void get_pixels(uint32_t *pixels);
        void handle_input(SDL_Event event, bool control, bool shift);
        void print_rules();
        void tick();
};
