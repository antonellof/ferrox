import { memo } from "react";
import {
  MarkdownTextPrimitive,
  unstable_memoizeMarkdownComponents as memoizeMarkdownComponents,
  type CodeHeaderProps,
} from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import { CopyButton } from "@/components/ui/copy-button";

// Rendering untrusted model output.
//
// The property the hand-written renderer had must survive the move to a
// library: a model's `<script>` has to reach the screen as the literal
// characters the model wrote, never as markup. `react-markdown` — which
// is what this primitive wraps — gives that BY CONSTRUCTION rather than
// by sanitising: it builds a React element tree from the mdast and has
// no raw-HTML path at all unless `rehype-raw` is added to the pipeline.
// It is not added here, and must not be. `remark-gfm` only adds tables,
// strikethrough, task lists and autolinks — no HTML.
//
// Links get the same treatment they had before: react-markdown's default
// `urlTransform` drops anything that is not a safe scheme, so a
// `javascript:` href in a model's answer renders inert. The `rel` below
// is belt-and-braces on top of that.

function CodeHeader({ language, code }: CodeHeaderProps) {
  return (
    // Same surface as the `<pre>` it sits on, so the two siblings read
    // as one code card rather than as a lid on a box.
    <div className="aui-code-header flex items-center gap-2 bg-sunken px-3 py-1.5">
      <span className="font-mono text-2xs tracking-wide text-faint uppercase">
        {language || "text"}
      </span>
      <span className="flex-1" />
      <CopyButton getText={() => code} label="Copy code" />
    </div>
  );
}

const components = memoizeMarkdownComponents({
  CodeHeader,
  a: (props) => (
    <a target="_blank" rel="noopener noreferrer nofollow" {...props} />
  ),
});

export const MarkdownText = memo(function MarkdownText() {
  return (
    <MarkdownTextPrimitive
      className="aui-md"
      remarkPlugins={[remarkGfm]}
      components={components}
      // Smooth reveal is assistant-ui's own animation over text it has
      // already received. It is a rendering rate, not a measurement, and
      // nothing downstream reads it as one.
      smooth
      defer
    />
  );
});
