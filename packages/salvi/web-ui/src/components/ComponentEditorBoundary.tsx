import { Component, type ErrorInfo, type ReactNode } from "react";
import { TriangleAlert } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class ComponentEditorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, information: ErrorInfo) {
    console.error("SALVI component editor failed", error, information);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="component-editor-error" role="alert">
          <TriangleAlert size={18} />
          <span>
            <strong>This parameter editor could not be displayed.</strong>
            <small>{this.state.error.message}</small>
          </span>
        </div>
      );
    }
    return this.props.children;
  }
}
