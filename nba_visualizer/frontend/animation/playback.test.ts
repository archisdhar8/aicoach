import { describe, expect, it } from 'vitest'
import { SimulationClock } from './playback'

describe('SimulationClock', () => {
  it('pauses and resumes without counting paused real time', () => {
    const clock = new SimulationClock(10)
    clock.play(1_000)
    expect(clock.advance(2_000)).toBe(1)
    clock.pause(2_500)
    expect(clock.timeSeconds).toBe(1.5)
    expect(clock.advance(8_000)).toBe(1.5)
    clock.play(8_000)
    expect(clock.advance(9_000)).toBe(2.5)
  })

  it('applies speed changes deterministically', () => {
    const clock = new SimulationClock(10)
    clock.play(0)
    clock.setPlaybackRate(2, 1_000)
    expect(clock.timeSeconds).toBe(1)
    expect(clock.advance(2_000)).toBe(3)
    clock.setPlaybackRate(0.5, 2_000)
    expect(clock.advance(4_000)).toBe(4)
  })

  it('stops at the exact simulation duration', () => {
    const clock = new SimulationClock(2)
    clock.play(0)
    expect(clock.advance(5_000)).toBe(2)
    expect(clock.isPlaying).toBe(false)
  })
})
