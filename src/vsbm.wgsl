const KERNEL_ITERATIONS: i32 = 5;
const CANVAS_SIZE = vec2f(1024, 1024);

struct Uniforms {
    origin: vec3f,
    _p1: f32,
    right: vec3f,
    _p2: f32,
    up: vec3f,
    _p3: f32,
    forward: vec3f,
    _p4: f32,
    screen_size: vec2f,
    len: f32,
    _p5: f32,
}

@group(0) @binding(0) var<uniform> ui: Uniforms;

fn pixel_to_ndc(pos: vec2f) -> vec2f {
    let ndc = (pos / CANVAS_SIZE * 2.0 - 1.0);
    return vec2f(ndc.x, -ndc.y);
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
    let pos = array(
        vec2f(-1, -1),
        vec2f(-1, 1),
        vec2f(1, 1),
        vec2f(1, 1),
        vec2f(1, -1),
        vec2f(-1, -1),
    );
    return vec4f(pos[idx], 0.0, 1.0);
}

fn kernel(ver: vec3f) -> f32 {
    var a = ver;
    for(var i: i32 = 0; i < KERNEL_ITERATIONS; i++) {
        var b = length(a);
        let c = atan2(a.y, a.x) * 8.0;
        let d = acos(a.z / b) * 8.0;
        b = b * b * b * b * b * b * b * b;
        a = vec3f(b * sin(d) * cos(c), b * sin(d) * sin(c), b * cos(d)) + ver;
        if (b > 6.0) { break; }
    }
    return 4.0 - dot(a, a);
}

@fragment
fn fs_main(@builtin(position) fs_pos: vec4f) -> @location(0) vec4f {
    let c = pixel_to_ndc(fs_pos.xy);
    let M_L = 0.381966;
    let M_R = 0.618033;
    let step_size = 0.002;

    let dir = ui.forward + ui.right * c.x * ui.screen_size.x + ui.up * c.y * ui.screen_size.y;
    let local_dir = normalize(vec3f(c.x * ui.screen_size.x, c.y * ui.screen_size.y, -1.0));

    var v1 = kernel(ui.origin + dir * (step_size * ui.len));
    var v2 = kernel(ui.origin);
    var sign = 0;
    var r3: f32 = 0.0;

    for (var k: i32 = 2; k < 1002; k++) {
        let ver = ui.origin + dir * (step_size * ui.len * f32(k));
        let v = kernel(ver);

        if (v > 0.0 && v1 < 0.0) {
            var r1 = step_size * ui.len * f32(k - 1);
            var r2 = step_size * ui.len * f32(k);
            for (var l = 0; l < 8; l++) {
                r3 = r1 * 0.5 + r2 * 0.5;
                if (kernel(ui.origin + dir * r3) > 0.0) { r2 = r3; } else { r1 = r3; }
            }
            if (r3 < 2.0 * ui.len) { sign = 1; break; }
        }

        if (v < v1 && v1 > v2 && v1 < 0.0 && (v1 * 2.0 > v || v1 * 2.0 > v2)) {
            var r1 = step_size * ui.len * f32(k - 2);
            var r2 = step_size * ui.len * (f32(k) - 2.0 + 2.0 * M_L);
            var r3_g = step_size * ui.len * (f32(k) - 2.0 + 2.0 * M_R);
            var r4 = step_size * ui.len * f32(k);
            var m2 = kernel(ui.origin + dir * r2);
            var m3 = kernel(ui.origin + dir * r3_g);

            for (var l = 0; l < 8; l++) {
                if (m2 > m3) {
                    r4 = r3_g; r3_g = r2; r2 = r4 * M_L + r1 * M_R;
                    m3 = m2; m2 = kernel(ui.origin + dir * r2);
                } else {
                    r1 = r2; r2 = r3_g; r3_g = r4 * M_R + r1 * M_L;
                    m2 = m3; m3 = kernel(ui.origin + dir * r3_g);
                }
            }

            let target_r = select(r3_g, r2, m2 > 0.0 || m3 > 0.0);
            if (kernel(ui.origin + dir * target_r) > 0.0) {
                var ra = step_size * ui.len * f32(k - 2);
                var rb = target_r;
                for (var l = 0; l < 8; l++) {
                    r3 = ra * 0.5 + rb * 0.5;
                    if (kernel(ui.origin + dir * r3) > 0.0) { rb = r3; } else { ra = r3; }
                }
                if (r3 < 2.0 * ui.len && r3 > step_size * ui.len) { sign = 1; break; }
            }
        }
        v2 = v1; v1 = v;
    }

    if (sign == 1) {
        let hit_pos = ui.origin + dir * r3;
        let r_sq = dot(hit_pos, hit_pos);
        let eps = r3 * 0.00025;

        var n: vec3f;
        n.x = kernel(hit_pos - ui.right * eps) - kernel(hit_pos + ui.right * eps);
        n.y = kernel(hit_pos - ui.up * eps) - kernel(hit_pos + ui.up * eps);
        n.z = kernel(hit_pos + ui.forward * eps) - kernel(hit_pos - ui.forward * eps);
        n = normalize(n);

        let reflect_v = reflect(local_dir, n);
        let light_dir = vec3f(0.276, 0.920, 0.276);

        let spec = pow(max(0.0, dot(reflect_v, light_dir)), 4.0);
        let diff = max(0.0, dot(n, light_dir));
        let shade = spec * 0.45 + diff * 0.25 + 0.3;

        let color = (sin(vec3f(r_sq * 10.0, r_sq * 10.0 + 2.05, r_sq * 10.0 - 2.05)) * 0.5 + 0.5) * shade;
        return vec4f(color, 1.0);
    } else {
        return vec4f(0.0);
    }
}
