"""Source binding and temporal preview contracts. Live/GPU parity is separate."""
from pathlib import Path
import importlib
import runpy
import struct
import sys
import types
import unittest

import numpy as np

package=types.ModuleType('_cloth_auto_test')
package.__path__=[str(Path(__file__).parents[1]/'pso2_tools')]
sys.modules[package.__name__]=package
files=importlib.import_module(package.__name__+'.native_cloth_file')
preview=importlib.import_module(package.__name__+'.native_cloth_preview')
fixture=runpy.run_path(str(Path(__file__).with_name('test_native_cloth_initialization.py')))['source_fixture']

class Estimates:
    rcp=staticmethod(lambda x:np.float32(1/np.float32(x)))
    rsqrt=staticmethod(lambda x:np.float32(1/np.sqrt(np.float32(x))))

def source():
    data=fixture()
    struct.pack_into('<I',data,0x104,0x460-32)
    struct.pack_into('<I',data,0x460,0x470-32)
    data[0x470:0x472]=b'a\0'
    return data

def world():
    value=np.tile(np.eye(4,dtype='f4'),(2,1,1))
    value[1,3,0]=1
    return value

def build(steps=5):
    return preview.FileClothPreview33(['a','drs-b'],files.read_parts(source(),['a','drs-b'],[-1,0]),
        world(),settle_steps=steps,seeds=Estimates)

class FileMappingTests(unittest.TestCase):
    def test_file_names_and_parents_generate_mapping_without_pose_data(self):
        part,=files.read_parts(source(),['a','drs-b'],[-1,0])
        np.testing.assert_array_equal(part.mapping,[0,1])
        np.testing.assert_array_equal(part.records,[[0,1,-1,-1,2]])
        self.assertEqual(part.definition.skip_mask,1)

    def test_missing_own_resource_is_not_mapped_by_name_similarity(self):
        with self.assertRaisesRegex(ValueError,'chain root is not in this model'):
            files.read_parts(source(),['a_similar','drs-b'],[-1,0])

    def test_count_mismatch_and_cycles_are_rejected(self):
        # A branch is walked (28C3490), so three drs particles exceed the two declared.
        for names,parents in [(['a','drs-b','drs-c'],[-1,0,0]),(['a','drs-b'],[1,0]),(['a','drs-b','drs-c'],[-1,0,1])]:
            with self.subTest(parents=parents),self.assertRaises(ValueError):
                files.read_parts(source(),names,parents)

    def test_branch_walk_uses_the_native_drs_name_filter(self):
        # First child: drs-/drs_/drs2-/drs2_; a later sibling only drs-.
        part,=files.read_parts(source(),['a','drs-b','drs_c'],[-1,0,0])
        np.testing.assert_array_equal(part.mapping,[0,1])
        part,=files.read_parts(source(),['a','hair_c','drs-b'],[-1,0,0])
        np.testing.assert_array_equal(part.mapping,[0,2])
        np.testing.assert_array_equal(part.records,[[0,1,-1,-1,2]])
        with self.assertRaisesRegex(ValueError,'declares 2 particles but this skeleton walks to 1'):
            files.read_parts(source(),['a','b'],[-1,0])

    def test_placeholder_fltd_without_cloth_is_empty_not_refused(self):
        data=source()
        head=files._File(data).header
        data[head+1]=data[head+2]=0
        self.assertFalse(files.declares_cloth(data))
        self.assertEqual(files.read_parts(data,['a','drs-b'],[-1,0]),())
        self.assertTrue(files.declares_cloth(source()))

    def test_unknown_shape_is_rejected_instead_of_dropped(self):
        data=source();data[0x102]=1
        struct.pack_into("<I",data,0x108,0x480-32);data[0x480]=9
        with self.assertRaisesRegex(ValueError,'collision'):
            files.read_parts(data,['a','drs-b'],[-1,0])

def with_sphere(bone):
    """One point sphere bound to `bone` (file kind 1, radius .1 at the origin)."""
    data=source();data[0x102]=1
    struct.pack_into('<I',data,0x108,0x480-32)
    data[0x480:0x480+92]=bytes(92);data[0x480]=1
    struct.pack_into('<4f',data,0x484,0,0,0,.1)
    struct.pack_into('<I',data,0x480+68,0x4f0-32)
    data[0x4f0:0x4f0+len(bone)+1]=bone.encode()+b'\0'
    return data

class RelatedResourceTests(unittest.TestCase):
    def test_bone_outside_the_model_resolves_in_the_parent_bank(self):
        # 28BFE30: own resource, then parent (bank 1), grandparent, siblings.
        # Found nowhere: registered with node -1, then 28BDC40 switches it off (0x10).
        lone,=files.read_parts(with_sphere('head'),['a','drs-b'],[-1,0])
        self.assertTrue(lone.colliders[0].disabled)
        self.assertEqual(lone.colliders[0].evaluate(world(),seeds=Estimates).flags & 0x11, 0x11)
        part,=files.read_parts(with_sphere('head'),['a','drs-b'],[-1,0],related={1:('character_top','head')})
        self.assertEqual(part.colliders[0].bindings,((1,1),))
        own,=files.read_parts(with_sphere('drs-b'),['a','drs-b'],[-1,0])
        self.assertEqual(own.colliders[0].bindings,(1,))

    def test_parent_bank_matrix_places_the_collider_like_an_own_bone(self):
        m=np.eye(4,dtype='f4');m[3,:3]=(.2,1.5,-.1)
        own_world=world();own_world[1]=m
        bank={1:np.stack([np.eye(4,dtype='f4'),m])}
        part,=files.read_parts(with_sphere('head'),['a','drs-b'],[-1,0],related={1:('character_top','head')})
        own,=files.read_parts(with_sphere('drs-b'),['a','drs-b'],[-1,0])
        a=part.colliders[0].evaluate(world(),seeds=Estimates,related_worlds=bank)
        b=own.colliders[0].evaluate(own_world,seeds=Estimates)
        np.testing.assert_array_equal(a.transform,b.transform)
        np.testing.assert_array_equal(a.inverse_transform,b.inverse_transform)
        self.assertEqual(a.flags,b.flags&~0x4000)

class CapsuleGrowthTests(unittest.TestCase):
    def test_flag_0x800_grows_end_radii_by_clamped_movement(self):
        # 28E5032: r * min(max(rate * moved + 1, 1), cap) per end.
        capsule=importlib.import_module(package.__name__+'.native_cloth_capsule')
        g=capsule.grow_capsule_radii(np.float32(.05),np.float32(.04),np.array([.9,0],'f4'),3,2.5)
        self.assertEqual(g,(np.float32(np.float32(2.5)*np.float32(.05)),np.float32(.04)))
        g=capsule.grow_capsule_radii(np.float32(.05),np.float32(.04),np.array([.1,.0],'f4'),2,1.5)
        self.assertAlmostEqual(float(g[0]),.05*1.2,places=6)


class ReferenceOverrideTests(unittest.TestCase):
    def test_type2_modifier_turns_the_bone_in_its_own_frame(self):
        # 2979150/54C250: M (Y 90 degrees, position 0) folds into the
        # reference pose; the bone stays put, its child swings about it.
        class Part:
            mapping = np.array([0, 1, 2])
            parents = (-1, 0, 1)
            definition = type('D', (), {'reference_overrides': ((1, (0, 0, 0), (0, 90, 0)),)})()
        ref = np.tile(np.eye(4, dtype='f4'), (3, 1, 1))
        ref[1, 3, :3] = (1, 0, 0)
        ref[2, 3, :3] = (2, 0, 0)
        out = preview.FileClothPreview33._override_reference(None, ref, Part())
        np.testing.assert_allclose(out[1, 3, :3], (1, 0, 0), atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(out[2, 3, :3] - out[1, 3, :3]), 1, atol=1e-5)
        self.assertGreater(abs(float(out[2, 3, 2])), .99)
        np.testing.assert_array_equal(out[0], ref[0])


class PreviewLifecycleTests(unittest.TestCase):
    def test_duplicate_evaluation_is_idempotent_and_source_is_preserved(self):
        sim=build();pose=world();original=pose.copy()
        a=sim.evaluate(pose,['a','drs-b'],1,60)
        state=sim.drivers[0].state.positions.copy()
        for _ in range(3):
            np.testing.assert_array_equal(sim.evaluate(pose,['a','drs-b'],1,60),a)
            np.testing.assert_array_equal(sim.drivers[0].state.positions,state)
        np.testing.assert_array_equal(pose,original)

    def test_forward_motion_updates_particles_and_reconstruction(self):
        sim=build();positions=[]
        for frame in range(1,31):
            pose=world();pose[:,3,1]+=np.float32(.1*np.sin(frame*.13))
            out=sim.evaluate(pose,['a','drs-b'],frame,60)
            np.testing.assert_allclose(out[:,3,:3],sim.drivers[0].state.positions[:,:3],atol=1e-6)
            positions.append(out[1,3,:3].copy())
        self.assertGreater(float(np.ptp(positions,axis=0).max()),.01)

    def test_jump_and_rewind_restart_from_current_pose_deterministically(self):
        sim=build();pose=world()
        sim.evaluate(pose,['a','drs-b'],1,60);sim.evaluate(pose,['a','drs-b'],2,60)
        for frame in [10,1]:
            out=sim.evaluate(pose,['a','drs-b'],frame,60)
            fresh=build().evaluate(pose,['a','drs-b'],frame,60)
            np.testing.assert_array_equal(out,fresh)
            self.assertEqual(sim.last_reset_reason,'timeline jump')

    def test_fps_change_reinitializes_instead_of_reusing_wrong_history(self):
        sim=build();sim.evaluate(world(),['a','drs-b'],1,60)
        out=sim.evaluate(world(),['a','drs-b'],2,30)
        np.testing.assert_array_equal(out,build().evaluate(world(),['a','drs-b'],2,30))
        self.assertEqual(sim.last_reset_reason,'FPS changed')

    def test_same_frame_initial_pose_edit_resettles_once(self):
        sim=build();sim.evaluate(world(),['a','drs-b'],1,60)
        moved=world();moved[:,3,0]+=.3
        out=sim.evaluate(moved,['a','drs-b'],1,60)
        np.testing.assert_array_equal(out,build().evaluate(moved,['a','drs-b'],1,60))
        np.testing.assert_array_equal(out,sim.evaluate(moved,['a','drs-b'],1,60))

    def test_wrapper_bypass_particles_follow_their_targets(self):
        # Part option bit 0 -> wrapper 0x20: no solver step (290A1D1), the
        # particles take the prepared targets (290A687), then writeback.
        from dataclasses import replace
        part,=files.read_parts(source(),['a','drs-b'],[-1,0])
        part=replace(part,definition=replace(part.definition,wrapper_flags=part.definition.wrapper_flags|0x20))
        sim=preview.FileClothPreview33(['a','drs-b'],(part,),world(),settle_steps=3,seeds=Estimates)
        for frame in range(1,6):
            pose=world();pose[:,3,1]+=np.float32(.1*frame)
            out=sim.evaluate(pose,['a','drs-b'],frame,60)
            np.testing.assert_allclose(out[:,3,:3],pose[:,3,:3],atol=1e-6)
            np.testing.assert_array_equal(sim.drivers[0].state.positions[:,:3],pose[:,3,:3])

    def test_actor_motion_is_in_simulation_space(self):
        sim=build();sim.evaluate(world(),['a','drs-b'],1,60)
        move=np.eye(4,dtype='f4');move[0,3]=.2
        sim.context(move,[0,-9.81,0,0]);out=sim.evaluate(world(),['a','drs-b'],2,60)
        self.assertAlmostEqual(float(sim.drivers[0].state.positions[0,0]),.2,places=6)
        self.assertAlmostEqual(float(out[0,3,0]),0,places=6)
        self.assertFalse(sim.live_dynamic_parity_verified)

    def test_actor_scale_change_rebuilds_reference_lengths(self):
        sim=build();sim.evaluate(world(),['a','drs-b'],1,60)
        scaled=np.eye(4,dtype='f4');scaled[0,0]=2
        sim.context(scaled,[0,-9.81,0,0]);out=sim.evaluate(world(),['a','drs-b'],2,60)
        other=build();other.context(scaled,[0,-9.81,0,0])
        np.testing.assert_array_equal(out,other.evaluate(world(),['a','drs-b'],2,60))
        self.assertEqual(sim.last_reset_reason,'actor scale changed')

    def _moving(self, frame, bump=0.0):
        w=world().copy();w[0,3,0]=np.float32(.03*np.sin(frame*.4)+bump);w[1,3,0]+=w[0,3,0]
        return w

    def test_history_replays_bitwise_and_resolves_changed_inputs(self):
        sim=build();first=[sim.evaluate(self._moving(f),['a','drs-b'],f,60) for f in range(1,21)]
        self.assertEqual(len(sim.history),20)
        state=sim.drivers[0].state
        # A timeline loop replays the stored frames exactly, without solving.
        again=[sim.evaluate(self._moving(f),['a','drs-b'],f,60) for f in range(1,21)]
        for a,b in zip(first,again):np.testing.assert_array_equal(a,b)
        # Continuing past the stored range steps on from the restored state.
        nxt=sim.evaluate(self._moving(21),['a','drs-b'],21,60)
        ref=build();[ref.evaluate(self._moving(f),['a','drs-b'],f,60) for f in range(1,21)]
        np.testing.assert_array_equal(nxt,ref.evaluate(self._moving(21),['a','drs-b'],21,60))
        # An edited frame re-solves from there and drops the later frames.
        sim.evaluate(self._moving(1),['a','drs-b'],1,60)
        edited=[sim.evaluate(self._moving(f,.01 if f>=10 else 0),['a','drs-b'],f,60) for f in range(2,16)]
        ref=build();ref.evaluate(self._moving(1),['a','drs-b'],1,60)
        expected=[ref.evaluate(self._moving(f,.01 if f>=10 else 0),['a','drs-b'],f,60) for f in range(2,16)]
        for a,b in zip(edited,expected):np.testing.assert_array_equal(a,b)
        self.assertEqual(max(sim.history),15)
        self.assertFalse(any(np.array_equal(edited[-1],x) for x in first))

    def test_rotating_scaled_actor_keeps_simulating(self):
        sim=build()
        def turned(angle):
            c,s=np.cos(angle),np.sin(angle)
            m=np.eye(4,dtype='f4');m[:3,:3]=np.float32(1.05536)*np.array([[c,-s,0],[s,c,0],[0,0,1]],'f4')
            return m
        sim.context(turned(0),[0,-9.81,0,0]);sim.evaluate(world(),['a','drs-b'],1,60)
        for frame in range(2,40):
            sim.context(turned(frame*.037),[0,-9.81,0,0]);sim.evaluate(world(),['a','drs-b'],frame,60)
            self.assertFalse(sim.settled_at_frame,frame)

    def test_nonfinite_input_fails_without_advancing_state(self):
        sim=build();sim.evaluate(world(),['a','drs-b'],1,60)
        before=sim.drivers[0].state.positions.copy();bad=world();bad[1,0,0]=np.nan
        with self.assertRaises(ValueError):sim.evaluate(bad,['a','drs-b'],2,60)
        np.testing.assert_array_equal(before,sim.drivers[0].state.positions)
        self.assertEqual(sim.frame,1)

if __name__=='__main__':unittest.main()
