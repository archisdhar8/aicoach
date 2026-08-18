import { expect, test, type Page } from '@playwright/test'

const HANDLER_ID = '10000000-0000-0000-0000-000000000001'
const RECEIVER_ID = '10000000-0000-0000-0000-000000000002'
const CUTTER_ID = '10000000-0000-0000-0000-000000000003'

test('create dribble, pass, cut, and shot without an action toolbar', async ({ page }) => {
  await page.goto('/')
  const court = page.getByTestId('basketball-court')
  const handler = page.locator(`[data-player-id="${HANDLER_ID}"]`)
  const receiver = page.locator(`[data-player-id="${RECEIVER_ID}"]`)
  const cutter = page.locator(`[data-player-id="${CUTTER_ID}"]`)
  await expect(court).toBeVisible()
  await expect(page.getByRole('group', { name: 'Draw basketball action' })).toHaveCount(0)

  if (await page.getByRole('button', { name: 'Clear', exact: true }).isEnabled()) {
    await page.getByRole('button', { name: 'Clear', exact: true }).click()
  }
  await clickPlayer(page, handler)
  const giveBall = page.getByRole('button', { name: 'Give ball to this player' })
  if (await giveBall.isVisible()) await giveBall.click()
  await page.getByRole('button', { name: 'Done positioning →' }).click()

  const start = await playerCourtPosition(handler)
  const dribbleTarget = { x: Math.min(start.x + 9, 86), y: Math.max(6, start.y - 5) }
  await dragToCourtPosition(page, court, handler, dribbleTarget)
  expect(await page.locator('[data-action-type="dribble"]').count()).toBeGreaterThan(0)

  await page.getByRole('button', { name: 'Add screen' }).click()
  await page.getByRole('button', { name: 'Use Screener as screener' }).click()
  expect(await page.locator('[data-action-type="screen"]').count()).toBeGreaterThan(0)
  expect(await page.locator('[data-defender-preview-id]').count()).toBeGreaterThan(0)
  await page.getByText('Advanced', { exact: true }).click()
  await expect(page.locator('.coverage-status')).toContainText('1 detected screen event')
  await page.getByText('Advanced', { exact: true }).click()

  await clickPlayer(page, handler)

  await clickPlayer(page, receiver)
  expect(await page.locator('[data-action-type="pass"]').count()).toBeGreaterThan(0)

  const cutterStart = await playerCourtPosition(cutter)
  await dragToCourtPosition(page, court, cutter, {
    x: Math.min(cutterStart.x + 8, 86),
    y: Math.min(cutterStart.y + 7, 46),
  })
  expect(await page.locator('[data-action-type="cut"]').count()).toBeGreaterThan(0)

  await clickPlayer(page, receiver)
  await page.getByRole('button', { name: 'Shoot at rim' }).click()
  expect(await page.locator('[data-action-type="shoot"]').count()).toBeGreaterThan(0)

  await page.getByRole('button', { name: 'Play', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeVisible({ timeout: 8_000 })
  await expect(handler).toHaveAttribute('data-player-x', dribbleTarget.x.toFixed(3))
  await expect(handler).toHaveAttribute('data-player-y', dribbleTarget.y.toFixed(3))
  await expect(page.locator('.shot-result-feedback')).toHaveCount(1)
})

async function clickPlayer(_page: Page, player: ReturnType<Page['locator']>): Promise<void> {
  await player.focus()
  await player.press('Enter')
}

async function playerCourtPosition(player: ReturnType<Page['locator']>): Promise<{ x: number; y: number }> {
  return {
    x: Number(await player.getAttribute('data-player-x')),
    y: Number(await player.getAttribute('data-player-y')),
  }
}

async function dragToCourtPosition(
  page: Page,
  court: ReturnType<Page['getByTestId']>,
  player: ReturnType<Page['locator']>,
  target: { x: number; y: number },
): Promise<void> {
  await player.scrollIntoViewIfNeeded()
  const courtBox = await court.boundingBox()
  const playerBox = await player.locator('circle').last().boundingBox()
  if (courtBox === null || playerBox === null) throw new Error('court or player is not visible')
  const targetPixels = {
    x: courtBox.x + ((target.x - 47) / 47) * courtBox.width,
    y: courtBox.y + (target.y / 50) * courtBox.height,
  }
  await page.mouse.move(playerBox.x + playerBox.width / 2, playerBox.y + playerBox.height / 2)
  await page.mouse.down()
  await page.mouse.move(targetPixels.x, targetPixels.y, { steps: 18 })
  await page.mouse.up()
}
