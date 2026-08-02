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
  const searchStageBox = await searchStage.boundingBox();
  const searchAddBox = await searchAdd.boundingBox();
  expect(searchStageBox).not.toBeNull();
  expect(searchAddBox).not.toBeNull();
  expect(searchAddBox!.y + searchAddBox!.height).toBeLessThanOrEqual(
    searchStageBox!.y + searchStageBox!.height + 1
  );
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

    await engineNode.click();
    await expect(page.locator(".drawer")).toBeVisible();
    await expect(page.locator(".drawer").getByText("1 configured")).toBeVisible();
    await expect(page.locator(".drawer .instance-card.active")).toHaveCount(1);
    await page.locator(".drawer-close").click();

    await page.getByRole("button", { name: "Add component in SEARCH" }).click();
    const nsga2 = page
      .locator(".drawer .instance-card")
      .filter({ hasText: "Pymoo Nsga2" });
    await nsga2.getByRole("button", { name: "Replace current" }).click();
    await expect(page.getByRole("button", { name: /Search engine: Pymoo Nsga2/i })).toBeVisible();
    await page.locator(".drawer-close").click();

    const invalidArchive = page.getByRole("button", { name: /Archive: Deep Grid Mome/i });
    await expect(invalidArchive).toHaveClass(/state-invalid/);
    await invalidArchive.click();
    await expect(page.locator(".drawer").getByRole("heading", { name: "Archive" })).toBeVisible();
    await expect(page.locator(".drawer textarea.code-input")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Build" })).toBeVisible();
    await page.locator(".drawer-close").click();

    const invalidObservers = page
      .locator(".workflow-node.state-invalid")
      .filter({ hasText: "Observers" });
    await expect(invalidObservers).toHaveCount(1);
    await invalidObservers.click();
    await expect(page.locator(".drawer").getByRole("heading", { name: "Observers" })).toBeVisible();
    await expect(
      page.locator(".drawer").getByText("archive_coverage still requires: archive")
    ).toBeVisible();
    await page.locator(".drawer-close").click();
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
