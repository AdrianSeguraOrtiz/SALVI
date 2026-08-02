import { useEffect, useRef } from "react";
import {
  BarChart,
  BoxplotChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
  type BoxplotSeriesOption,
  type HeatmapSeriesOption,
  type ScatterSeriesOption
} from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  type DataZoomComponentOption,
  type GridComponentOption,
  type LegendComponentOption,
  type TooltipComponentOption,
  type VisualMapComponentOption
} from "echarts/components";
import * as echarts from "echarts/core";
import type { BarSeriesOption, LineSeriesOption } from "echarts/charts";
import { CanvasRenderer } from "echarts/renderers";

type ChartOption = echarts.ComposeOption<
  | BarSeriesOption
  | BoxplotSeriesOption
  | HeatmapSeriesOption
  | LineSeriesOption
  | ScatterSeriesOption
  | DataZoomComponentOption
  | GridComponentOption
  | LegendComponentOption
  | TooltipComponentOption
  | VisualMapComponentOption
>;

echarts.use([
  BarChart,
  BoxplotChart,
  CanvasRenderer,
  DataZoomComponent,
  GridComponent,
  HeatmapChart,
  LegendComponent,
  LineChart,
  ScatterChart,
  TooltipComponent,
  VisualMapComponent
]);

interface Props {
  option: ChartOption;
  height: number;
}

export function Chart({ option, height }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const pendingOption = useRef<ChartOption | null>(null);
  const interacting = useRef(false);
  const legendScrollIndex = useRef(0);

  useEffect(() => {
    if (!container.current) return;
    const instance = echarts.init(container.current, undefined, { renderer: "canvas" });
    chart.current = instance;
    instance.on("legendscroll", (event: unknown) => {
      if (
        typeof event === "object" &&
        event !== null &&
        "scrollDataIndex" in event &&
        typeof event.scrollDataIndex === "number"
      ) {
        legendScrollIndex.current = event.scrollDataIndex;
      }
    });
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(container.current);
    return () => {
      observer.disconnect();
      instance.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    const instance = chart.current;
    if (!instance) return;
    if (interacting.current) {
      pendingOption.current = option;
      return;
    }
    updateChart(instance, option, legendScrollIndex.current);
  }, [option]);

  function finishInteraction() {
    interacting.current = false;
    const instance = chart.current;
    const next = pendingOption.current;
    if (!instance || !next) return;
    pendingOption.current = null;
    updateChart(instance, next, legendScrollIndex.current);
  }

  return (
    <div
      className="chart"
      style={{ height }}
      ref={container}
      onPointerEnter={() => {
        interacting.current = true;
      }}
      onPointerLeave={finishInteraction}
    />
  );
}

function updateChart(instance: echarts.ECharts, option: ChartOption, scrollDataIndex: number) {
  instance.setOption(option, {
    notMerge: false,
    lazyUpdate: true,
    replaceMerge: ["series"]
  });
  if (scrollDataIndex > 0) {
    instance.dispatchAction({
      type: "legendScroll",
      scrollDataIndex
    });
  }
}
