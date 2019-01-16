macro_rules! ais_frame{
    ($()) => {}
}

ais_frame!{NavigationBlock,
    {
        type[6]     : u(u8),
        repeat[2]   : u(u8),
        mmsi[30]    : u(u32),
        status[4]   : u(u8),
        turn[8]     : I(3, f32),
        speed[10]   : U(1, f32),
        accuracy[1] : b(),
        lon[28]     : I(4, f32),
        lat[27]     : I(4, f32),
        course[12]  : U(1, f32),
        heading[9]  : u(u32),
        second[6]   : u(u8),
        maneuver[2] : u(u8),
        _[3]        : x(),
        raim[1]     : b(),
        radio[19]   : u(u32),
    }
}
