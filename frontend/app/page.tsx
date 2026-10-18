import Chat from "@/components/Chat";
import Ingest from "@/components/Ingest";
import KeyPanel from "@/components/KeyPanel";

export default function WorkspacePage() {
  return (
    <div className="workspace">
      <aside className="rail">
        <KeyPanel />
        <Ingest />
      </aside>
      <section className="main">
        <Chat />
      </section>
    </div>
  );
}
