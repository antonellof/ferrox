import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import {
  createBrowserRouter,
  Navigate,
  RouterProvider,
} from "react-router";
import { AppShell } from "@/components/app-shell";
import { ChatScreen } from "@/screens/chat";
import { ModelsScreen } from "@/screens/models";
import { ActivityScreen } from "@/screens/activity";
import { ConnectScreen } from "@/screens/connect";
import "@/index.css";

// `/` and `/ui` both land on Chat; `/ui/<screen>` deep links resolve
// directly. A static host answers every one of these paths with the same
// `index.html`, so a reload or a bookmark lands on the right screen
// instead of a 404.
//
// **The conversation is in the URL.** `/ui/chat` is a new chat and
// `/ui/chat/<id>` is that conversation, which is what makes the base URL
// a predictable entry point rather than "whatever was open last" — see
// `lib/entry-state.ts` for the whole rule. The optional segment is one
// route, not two, on purpose: two route objects would remount
// `ChatScreen` on every switch between a new chat and a saved one, and
// remounting it throws away the runtime holding the thread.
const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/ui/chat" replace /> },
      { path: "ui", element: <Navigate to="/ui/chat" replace /> },
      { path: "ui/chat/:conversationId?", element: <ChatScreen /> },
      { path: "ui/models", element: <ModelsScreen /> },
      { path: "ui/activity", element: <ActivityScreen /> },
      { path: "ui/connect", element: <ConnectScreen /> },
      // Anything else the SPA fallback handed us is not a screen.
      { path: "*", element: <Navigate to="/ui/chat" replace /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
