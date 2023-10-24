import numpy as np
from manimlib import *
from itertools import permutations


base_face = lambda: [
    Dot(radius=0.5, fill_color=rgb2hex((0.9, 0.9, 0)), fill_opacity=1),
    Dot(radius=0.1, fill_color=BLACK).move_to(LEFT * 0.25 + UP * 0.17),
    Dot(radius=0.1, fill_color=BLACK).move_to(RIGHT * 0.25 + UP * 0.17),
]

patches = {
    "smile": lambda: ArcBetweenPoints(LEFT * 0.3 + DOWN * 0.15, RIGHT * 0.3 + DOWN * 0.15, angle=TAU / 5, color=BLACK),
    "frown": lambda: ArcBetweenPoints(LEFT * 0.3 + DOWN * 0.25, RIGHT * 0.3 + DOWN * 0.25, angle=-TAU / 5, color=BLACK),
    "left_eb_down": lambda: Line(LEFT * 0.15 + UP * 0.3, LEFT * 0.35 + UP * 0.35, color=BLACK),
    "left_eb_up": lambda: Line(LEFT * 0.15 + UP * 0.35, LEFT * 0.35 + UP * 0.3, color=BLACK),
    "right_eb_down": lambda: Line(RIGHT * 0.15 + UP * 0.3, RIGHT * 0.35 + UP * 0.35, color=BLACK),
    "right_eb_up": lambda: Line(RIGHT * 0.15 + UP * 0.35, RIGHT * 0.35 + UP * 0.3, color=BLACK),
}


def make_face(names):
    return VGroup(*(base_face() + [patches[i]() for i in names]))


sorted_keys = sorted(patches.keys())


def has_overlap(face):
    # Can't have two overlapping eyebrows!
    if "left_eb_down" in face and "left_eb_up" in face:
        return True
    if "right_eb_down" in face and "right_eb_up" in face:
        return True
    # Can't have two overlapping mouths!
    if "smile" in face and "frown" in face:
        return True
    return False


def face_reward(face):
    if has_overlap(face):
        return 0
    eyebrows = "left_eb_down", "left_eb_up", "right_eb_down", "right_eb_up"
    # Must have exactly two eyebrows
    if sum([i in face for i in eyebrows]) != 2:
        return 0
    # We want twice as many happy faces as sad faces so here we give a reward of 2 for smiles
    if "smile" in face:
        return 2
    if "frown" in face:
        return 1  # and a reward of 1 for frowns
    # If we reach this point, there's no mouth
    return 0


def face_hash(face):
    return tuple([i in face for i in sorted_keys])


def face_mdp():
    enumerated_states = []
    transitions = []

    def recursively_enumerate(s):
        if has_overlap(s):
            return
        for i in sorted_keys:
            if i not in s:
                recursively_enumerate(s + [i])
        enumerated_states.append(s)
        if len(s):
            transitions.append((s[:-1], s))

    recursively_enumerate([])
    unique = []
    for i in map(set, enumerated_states):
        if i not in unique:
            unique.append(i)
    enumerated_states = sorted([tuple(sorted(i)) for i in unique])

    transitions = sorted(set([(tuple(sorted(a)), tuple(sorted(b))) for a, b in transitions]))

    lens = [len([i for i in enumerated_states if len(i) == j]) for j in range(4)]
    levels = [sorted([i for i in enumerated_states if len(i) == j]) for j in range(4)]

    face2pos = {}
    for i, (level, L) in enumerate(zip(levels, lens)):
        for j, face in enumerate(level):
            face2pos[face_hash(face)] = (j / L + 0.5 / L, i / 4, i)
    arrows = []
    num_parents = {}
    for a, b in transitions:
        ah = face_hash(a)
        bh = face_hash(b)
        num_parents[bh] = num_parents.get(bh, 0) + 1
        if not len(b):
            continue
        arrows.append((ah, bh))

    def compute_flows(rewards):
        terminals = levels[-1]
        flows = {}
        for t, r in zip(terminals, rewards):
            flows[face_hash(t)] = r
        for l in levels[:-1][::-1]:
            for s in l:
                sh = face_hash(s)
                for a, b in transitions:
                    ah = face_hash(a)
                    bh = face_hash(b)
                    if sh == ah:
                        flows[sh] = flows.get(sh, 0) + flows[bh] / num_parents[bh]
                        flows[(ah, bh)] = flows[bh] / num_parents[bh]
        return flows

    return face2pos, arrows, compute_flows


DOT_DENSITY = 20
from rdkit import Chem
from rdkit.Chem import Draw


class Smileys(Scene):
    is_mols = False

    def construct(self) -> None:
        self.camera.background_rgba = [1, 1, 1, 1]

        face2pos, arrows, self.compute_flows = face_mdp()
        self.faces = faces = []
        self.i2pos = i2pos = lambda i: np.float32((face2pos[i][1] * 12 - 6 + 1.5, face2pos[i][0] * 7 - 3.5, 0))
        self.face2obj = {}
        frags = ["C1CC1", "c1ccncc1", "S", "CCNCC", "c1cncnc1", "C1COCCC1"]
        asd = 0
        empty = Draw.MolToImage(Chem.MolFromSmiles(""), (100, 90), fitImage=True)
        for i in face2pos:
            if self.is_mols:
                smi = "".join(f for f, b in zip(frags, i) if b) or "C"
                img = Draw.MolToImage(Chem.MolFromSmiles(smi), (100, 90), fitImage=True)
                if asd == 15:
                    # Why? IDK the Manim gods are angry (haha Copilot suggestion)
                    # seriously though if I don't do this, the image is rendered as a red square or some other object
                    # steals this MObject's texture
                    # and also this https://github.com/3b1b/manim/pull/2055 was necessary for images to work
                    # properly for some reason
                    f = []
                    f.append(ImageMobject("", height=1, image=img))
                    for _ in range({2: 4, 3: 3, 10: 4}.get(DOT_DENSITY, 4)):
                        f.append(ImageMobject("", height=1, image=img))
                    # f.append(ImageMobject("", height=1, image=empty))
                    faces.append(Group(*f))
                else:
                    faces.append(ImageMobject("", height=1, image=img))
                asd += 1
            else:
                faces.append(make_face([sorted_keys[j] for j in range(len(sorted_keys)) if i[j]]))
            faces[-1].move_to(i2pos(i)).scale([1, 0.9, 0.52, 0.65][face2pos[i][2]])
            self.face2obj[i] = faces[-1]
        self.arrow_lines = arrow_lines = []
        self.fh2a = {}
        for a, b in arrows:
            arrow_lines.append(
                VGroup(
                    Arrow(
                        self.face2obj[a].get_right(),
                        self.face2obj[b].get_left(),
                        stroke_color=BLACK,
                        tip_width_ratio=4,
                        buff=0.05,
                    ),
                    Arrow(
                        self.face2obj[a].get_right(),
                        self.face2obj[b].get_left(),
                        stroke_color=GREY,
                        tip_width_ratio=4,
                        buff=0.05,
                        stroke_width=1.5,
                    ),
                )
            )
            self.fh2a[(a, b)] = arrow_lines[-1]
        self.construct_()

    def construct_(self):
        t = ["left_eb_down", "smile", "right_eb_down"]
        anims = []
        for i in range(4):
            anims.append(ShowCreation(self.face2obj[face_hash(t[:i])]))
            if i > 0:
                anims.append(ShowCreation(self.fh2a[(face_hash(t[: i - 1]), face_hash(t[:i]))]))
        self.play(*anims)
        self.wait(3)
        t = ["right_eb_down", "left_eb_down", "smile"]
        anims = []
        for i in range(4):
            o = self.face2obj[face_hash(t[:i])]
            if o not in self.mobjects:
                anims.append(ShowCreation(o))
            if i > 0:
                anims.append(ShowCreation(self.fh2a[(face_hash(t[: i - 1]), face_hash(t[:i]))]))
        self.play(*anims)
        self.wait(3)
        anims = []
        created = set()
        for t in permutations(["right_eb_down", "left_eb_down", "smile"], 3):
            for i in range(4):
                o = self.face2obj[face_hash(t[:i])]
                if o not in self.mobjects and id(o) not in created:
                    anims.append(ShowCreation(o))
                    created.add(id(o))
                if i > 0:
                    o = self.fh2a[(face_hash(t[: i - 1]), face_hash(t[:i]))]
                    if o not in self.mobjects and id(o) not in created:
                        created.add(id(o))
                        anims.append(ShowCreation(o))
        self.play(*anims)
        self.wait(3)
        self.play(
            ShowCreation(VGroup(*[i for i in self.faces + self.arrow_lines if i not in self.mobjects]), lag_ratio=0.01),
            run_time=5,
        )


class SmileysWRewards(Smileys):
    def construct_(self):
        rewards = [1, 1, 2, 3, 1, 0.5, 0.5, 3]
        self.add(*(self.faces + self.arrow_lines))
        reward_labels = [Tex(f"R={i}", font_size=48, color=BLACK) for i in rewards]
        for i, r in enumerate(reward_labels):
            r.move_to(self.faces[-i - 1].get_right() + r.get_width() / 2 * RIGHT + RIGHT * 0.1)
        self.play(*[ShowCreation(i) for i in reward_labels], run_time=2)
        self.wait(1)
        objs_to_keep = set()
        edges = []
        edge2obj = {}
        for t in permutations(["right_eb_down", "left_eb_down", "smile"], 3):
            for i in range(4):
                o = self.face2obj[face_hash(t[:i])]
                o.zindex = 2
                objs_to_keep.add(id(o))
                if i > 0:
                    u, v = face_hash(t[: i - 1]), face_hash(t[:i])
                    o = self.fh2a[(u, v)]
                    edges.append((u, v))
                    objs_to_keep.add(id(o))
                    o.zindex = 0
                    edge2obj[(u, v)] = o
        self.play(
            AnimationGroup(
                *[i.animate.set_opacity(0.1) for i in self.faces + self.arrow_lines if id(i) not in objs_to_keep]
            ),
            run_time=2,
        )
        rewards_just_s3 = [0, 0, 0, 3, 0, 0, 0, 0]
        flows = self.compute_flows(rewards_just_s3[::-1])
        streams = ParticleStreamGroup(edges, self.face2obj, flows, density=DOT_DENSITY)
        flow_labels = []
        for u, v in edges:
            o = edge2obj[(u, v)]
            angle = o.submobjects[0].get_angle()
            lab = Tex(f"F={flows[(u, v)]:.1f}", font_size=30, color=DARK_BROWN)
            lab.rotate(angle)
            lab.move_to(o.get_center() + rotate_vector(UP * 0.2, angle))
            flow_labels.append(lab)
        self.add(*streams.particles)
        self.mobjects = sorted(self.mobjects, key=lambda i: i.zindex if hasattr(i, "zindex") else 0)
        self.play(*streams.animate(1), *[ShowCreation(i) for i in flow_labels], run_time=1)
        self.play(*streams.animate(5), run_time=5)


class SmileysFull(SmileysWRewards):
    def construct_(self):
        rewards = [1, 1, 2, 3, 1, 0.5, 0.5, 3]
        self.add(*(self.faces + self.arrow_lines))
        reward_labels = [Tex(f"R={i}", font_size=48, color=BLACK) for i in rewards]
        for i, r in enumerate(reward_labels):
            r.move_to(self.faces[-i - 1].get_right() + r.get_width() / 2 * RIGHT + RIGHT * 0.1)
        self.add(*reward_labels)
        objs_to_keep = set()
        edges = []
        edge2obj = {}
        for t in permutations(["right_eb_down", "left_eb_down", "smile"], 3):
            for i in range(4):
                o = self.face2obj[face_hash(t[:i])]
                o.zindex = 2
                objs_to_keep.add(id(o))
                if i > 0:
                    u, v = face_hash(t[: i - 1]), face_hash(t[:i])
                    o = self.fh2a[(u, v)]
                    edges.append((u, v))
                    objs_to_keep.add(id(o))
                    o.zindex = 0
                    edge2obj[(u, v)] = o
        [i.set_opacity(0.1) for i in self.faces + self.arrow_lines if id(i) not in objs_to_keep]
        rewards_just_s3 = [0, 0, 0, 3, 0, 0, 0, 0]
        flows = self.compute_flows(rewards_just_s3[::-1])
        flow_labels = []
        for u, v in edges:
            o = edge2obj[(u, v)]
            angle = o.submobjects[0].get_angle()
            lab = Tex(f"F={flows[(u, v)]:.1f}", font_size=30, color=DARK_BROWN)
            lab.rotate(angle)
            lab.move_to(o.get_center() + rotate_vector(UP * 0.2, angle))
            flow_labels.append(lab)
        self.mobjects = sorted(self.mobjects, key=lambda i: i.zindex if hasattr(i, "zindex") else 0)
        flows = self.compute_flows(rewards[::-1])
        streams = ParticleStreamGroup(
            list(self.fh2a.keys()), self.face2obj, flows, density=DOT_DENSITY, pipe_radius=0.04
        )
        self.play(
            *streams.animate(1),
            *[FadeOut(i) for i in flow_labels],
            *[i.animate.set_opacity(1) for i in self.faces],
            *[i.animate.set_opacity(0.05) for i in self.arrow_lines],
            run_time=1,
        )
        self.play(*streams.animate(10), run_time=10)


class MolsFrags(SmileysFull):
    is_mols = True

    def construct_(self):
        rewards = [1, 1, 2, 3, 1, 0.5, 0.5, 3]
        self.add(*(self.faces + self.arrow_lines))
        reward_labels = [Tex(f"R={i}", font_size=48, color=BLACK) for i in rewards]
        for i, r in enumerate(reward_labels):
            r.move_to(self.faces[-i - 1].get_right() + r.get_width() / 2 * RIGHT + RIGHT * 0.1)
        self.add(*reward_labels)
        self.mobjects = sorted(self.mobjects, key=lambda i: i.zindex if hasattr(i, "zindex") else 0)
        flows = self.compute_flows(rewards[::-1])
        streams = ParticleStreamGroup(
            list(self.fh2a.keys()), self.face2obj, flows, density=DOT_DENSITY, pipe_radius=0.04
        )
        [i.set_opacity(0.05) for i in self.arrow_lines],
        self.play(
            *streams.animate(2),
            *[FadeOut(i) for i in reward_labels],
            run_time=2,
        )
        self.play(*streams.animate(10), run_time=10)


class MolsFragsSmall(Scene):
    is_mols = True

    def construct(self):
        self.camera.background_rgba = [1, 1, 1, 1]
        img1 = ImageMobject("", height=1, image=Draw.MolToImage(Chem.MolFromSmiles("C1CC1"), (160, 180), fitImage=True))
        img2 = ImageMobject(
            "", height=2, image=Draw.MolToImage(Chem.MolFromSmiles("C1CC1c1cncnc1"), (200, 180), fitImage=True)
        )
        img1.move_to(LEFT * 2.5)
        img2.move_to(RIGHT * 2.5)
        stream = ParticleStream(100, img1.get_right(), img2.get_left(), pipe_radius=0.04)
        arr = Arrow(img1.get_right(), img2.get_left(), tip_width_ratio=4, buff=0.05, stroke_color=BLACK)
        flabel = Tex("F_\\theta(s \\to s') = ", font_size=40, color=BLACK).move_to(arr.get_center() + DOWN * 0.4)
        flstream = ParticleStream(20, flabel.get_right() + RIGHT * 0.1, flabel.get_right() + RIGHT * 0.5)
        arr.set_opacity(0.5)
        other_arrs = (
            [
                Arrow(
                    img1.get_right() + UP * i * 0.05,
                    img2.get_left() + UP * i,
                    tip_width_ratio=4,
                    buff=0.05,
                    stroke_color=BLACK,
                ).set_opacity(0.05)
                for i in range(-5, 6)
            ]
            + [
                Arrow(
                    img1.get_left() + (img1.get_right() - img2.get_left()) + UP * i,
                    img1.get_left() + UP * i * 0.05,
                    tip_width_ratio=4,
                    buff=0.05,
                    stroke_color=BLACK,
                ).set_opacity(0.05)
                for i in range(-5, 6)
            ]
            + [
                Arrow(
                    img2.get_right() + UP * i * 0.05,
                    img2.get_right() + UP * i + RIGHT * 4,
                    tip_width_ratio=4,
                    buff=0.05,
                    stroke_color=BLACK,
                ).set_opacity(0.05)
                for i in range(-5, 6)
            ]
        )
        self.add(img1, img2, arr, flabel, *other_arrs, *stream.particles, *flstream.particles)
        self.play(*stream.get_animations(6), *flstream.get_animations(6), run_time=6)


class ParticleStreamGroup:
    def __init__(self, edges, states, flows, density=10, pipe_radius=0.075):
        self.streams = []
        for i in range(len(edges)):
            u, v = edges[i]
            left, right = states[u].get_right(), states[v].get_left()
            self.streams.append(
                # To get the illusion right, we need to put more particles when the "pipe" is longer
                ParticleStream(
                    int(flows[(u, v)] * density * np.linalg.norm(left - right)), left, right, pipe_radius=pipe_radius
                )
            )
        self.particles = sum((i.particles for i in self.streams), [])
        for i in self.particles:
            i.zindex = 1

    def animate(self, duration):
        return sum((i.get_animations(duration) for i in self.streams), [])


class TheOriginalAnim(Scene):
    def construct(self):
        self.camera.background_rgba = [1, 1, 1, 1]
        nodes = np.float32(
            [
                [0, 2],
                [4, 0],
                [4, 4],
                [8, 0],
                [10, 3],
                [8, 5],
                [10, 8],
                [12, 0],
                [12, 5],
                [14, 8],
                [16, 0],
                [16, 3],
                [18, 6],
                [20, 1.5],
                [22, 4.5],
                [16, 6.5],
                [20, 9],
            ]
        )
        nodes *= 0.8 * 0.5  # determined by science
        nodes[:, 0] *= 1.2
        nodes[:, 0] += 1.5

        edges = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [3, 4],
            [2, 5],
            [5, 6],
            [3, 7],
            [5, 8],
            [8, 9],
            [7, 10],
            [7, 11],
            [8, 11],
            [11, 12],
            [10, 13],
            [11, 13],
            [13, 14],
            [8, 15],
            [15, 16],
        ]
        edgeflows = [2, 3, 2, 1, 1.5, 2, 0.5, 1.5, 1.5, 1, 1, 0.5, 0.2, 0.3, 1, 0.4, 1.4, 0.3, 0.3]
        is_terminal = [True] * len(nodes)
        for u, v in edges:
            is_terminal[u] = False
        is_terminating = [False] * len(nodes)
        for u, v in edges:
            is_terminating[u] = is_terminal[v]
            if is_terminal[v]:
                nodes[v] = nodes[u] + (0.8, 1)

        states = [
            Square(0.6, fill_color=BLUE, fill_opacity=1)
            if is_terminal[i]
            else (Dot(fill_color=BLUE, radius=0.3) if i > 0 else Triangle(color="#6677CC", fill_opacity=1).scale(0.4))
            for i in range(len(nodes))
        ]
        states[0].rotate(-90 * DEGREES)
        [i.scale(1.2) for i in states]  # Also determined by science
        state_labels = [
            Tex(f"s_{{{i}}}" if not is_terminal[i] else "s_f", font_size=35, color=BLACK) for i in range(len(nodes))
        ]

        for i, (s, sl) in enumerate(zip(states, state_labels)):
            s.move_to(np.float32((nodes[i][0], -nodes[i][1], 0)))
            if i == 0:
                sl.move_to(s.get_center_of_mass())
            else:
                sl.move_to(s.get_center())
        group = VGroup(*(states + state_labels))
        group.move_to(np.float32((0, 0, 0))).scale(1.15)

        streams = []
        for i in range(len(edges)):
            u, v = edges[i]
            streams.append(ParticleStream(int(edgeflows[i] * 50), states[u].get_center(), states[v].get_center()))

        self.add(*sum((i.particles for i in streams), []))
        self.add(group)
        self.play(*sum((i.get_animations(5) for i in streams), []), run_time=5)
        if 0:
            self.play(
                *sum((i.get_animations(4) for i in streams), []),
                self.camera.frame.animate.move_to(states[2].get_center()).scale(0.25),
                run_time=4,
            )


class ParticleStream:
    def __init__(self, n, start, end, speed=1, pipe_radius=0.075):
        self.n = n
        self.start = start
        self.end = end
        u = np.random.uniform
        self.particles = [Dot(fill_color=rgb2hex((u(0.1, 0.3), u(0.3, 0.7), u(0.5, 1))), radius=0.02) for i in range(n)]
        self.offsets = np.random.uniform(0, 1, n)
        self.pipe_offsets = np.random.uniform(-1, 1, n) * pipe_radius
        # The normalized vector from start to end, rotated 90 degrees
        self.pipe_vector = np.array([end[1] - start[1], start[0] - end[0], 0])
        self.pipe_vector /= np.linalg.norm(self.pipe_vector)
        self._last_duration = 0
        self.speed = speed / np.linalg.norm(end - start)

    def get_animations(self, duration):
        def get_interp(i):
            # Note, we need to use a closure here to capture the value of i
            return lambda x: (self.offsets[i] + x * duration * self.speed) % 1

        for i, p in enumerate(self.particles):
            p.move_to(self.start + self.pipe_vector * self.pipe_offsets[i])
        self.offsets = self.offsets + self._last_duration * self.speed
        self._last_duration = duration
        return [
            ApplyMethod(p.move_to, self.end + self.pipe_vector * self.pipe_offsets[i], rate_func=get_interp(i))
            for i, p in enumerate(self.particles)
        ]


if __name__ == "__main__":
    SmileysFull().run()
