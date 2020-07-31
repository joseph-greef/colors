
#include "initializer.h"
#include "rainbows.h"
#include "ruleset.h"


class LifeLike : public Ruleset {
    private:
        int _alive_color_scheme;
        int _alive_offset;
        int *_board;
        int *_board_buffer;
        bool _born[9];
        int _dead_color_scheme;
        int _dead_offset;
        Initializer _initializer;
        int _num_faders;
        Rainbows _rainbows;
        bool _stay_alive[9];

    public:
        LifeLike(int width, int height);
        ~LifeLike();
        void tick();
        void get_pixels(uint32_t *pixels);
};
