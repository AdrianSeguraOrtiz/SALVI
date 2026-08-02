import { expect, test } from "@playwright/test";

test("Build, Monitor, and Results remain usable without overlap", async ({ page }, testInfo) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Build" })).toBeVisible();
  await expect(
    page.getByRole("button", { name: /(?:Select|Change) input in INPUT/ })
  ).toBeVisible();
  await expect(page.getByText("Search engine", { exact: true })).toBeVisible();
  await expect(page.locator(".workflow-stage")).toHaveCount(6);
  await expect(page.locator(".workflow-add-node")).toHaveCount(6);
  expect(await page.locator(".workflow-node.state-configured").count()).toBeGreaterThan(5);
  await expect(page.getByText("Source filters", { exact: true })).toHaveCount(0);
  const searchStage = page.locator(".workflow-stage.stage-search");
  const searchAdd = page.getByRole("button", { name: "Add component in SEARCH" });
  await expect
    .poll(async () => {
      const searchStageBox = await searchStage.boundingBox();
      const searchAddBox = await searchAdd.boundingBox();
      if (!searchStageBox || !searchAddBox) return false;
      return (
        searchAddBox.y + searchAddBox.height <=
        searchStageBox.y + searchStageBox.height + 1
      );
    })
    .toBe(true);
  await expect
    .poll(() =>
      page
        .locator(".workflow-canvas")
        .evaluate((element) => element.scrollHeight > element.clientHeight)
    )
    .toBe(true);
  await searchAdd.scrollIntoViewIfNeeded();
  await expect(searchAdd).toBeVisible();

  if (testInfo.project.name === "desktop") {
    const engineNode = page.getByRole("button", { name: /Search engine: Serial MOME/i });
    const stageBox = await searchStage.boundingBox();
    const engineBox = await engineNode.boundingBox();
    expect(stageBox).not.toBeNull();
    expect(engineBox).not.toBeNull();
    expect(engineBox!.x).toBeGreaterThan(stageBox!.x);
    expect(engineBox!.x + engineBox!.width).toBeLessThan(stageBox!.x + stageBox!.width);

    await page.getByRole("button", { name: "Add component in PREPARATION" }).click();
    await expect(page.locator(".drawer")).toBeVisible();
    await expect(page.locator(".drawer").getByRole("heading", { name: "Preparation" })).toBeVisible();
    await expect(page.locator(".drawer").getByText("Configured", { exact: true })).toBeVisible();
    const sourceFilter = page
      .locator(".drawer .instance-card")
      .filter({ hasText: "Drop All Missing Columns" });
    await sourceFilter.getByRole("button", { name: "Use instance" }).click();
    const sourceFilterNode = page
      .locator(".workflow-node")
      .filter({ has: page.locator(".node-role", { hasText: "Source filters" }) });
    await expect(sourceFilterNode.locator(".node-instance")).toHaveText("1 selected");
    await page.locator(".drawer-close").click();

    await page.getByRole("radio", { name: /Conventional multi-objective/i }).click();
    await expect(page.getByRole("button", { name: /Search engine: Pymoo Nsga2/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Descriptors:/i })).toHaveCount(0);
    await expect(page.getByRole("button", { name: /Archive:/i })).toHaveCount(0);
    await page.getByRole("button", { name: /Search engine: Pymoo Nsga2/i }).click();
    const mome = page.locator(".drawer .instance-card").filter({ hasText: "Serial Mome" });
    await expect(mome).toHaveClass(/blocked/);
    await expect(mome.getByText(/belongs to the QUALITY_DIVERSITY search family/)).toBeVisible();
    await page.locator(".drawer-close").click();

    await page.getByRole("radio", { name: /Quality diversity/i }).click();
    await expect(page.getByRole("button", { name: /Search engine: Serial MOME/i })).toBeVisible();
  }

  await page.getByRole("button", { name: "Monitor" }).click();
  await expect(page.getByRole("heading", { name: "Monitor" })).toBeVisible();

  await page.getByRole("button", { name: "Results" }).click();
  await expect(page.getByRole("heading", { name: "Results" })).toBeVisible();

  const body = page.locator("body");
  const box = await body.boundingBox();
  expect(box?.width).toBeGreaterThan(0);
  expect(await page.locator("body").evaluate((node) => node.scrollWidth)).toBeLessThanOrEqual(
    await page.locator("body").evaluate((node) => node.clientWidth + 1)
  );
});
