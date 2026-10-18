import Chat from "@/components/Chat";
import Ingest from "@/components/Ingest";

export default function WorkspacePage() {
  return (
    <div className="workspace">
      <aside className="rail">
        <Ingest />
      </aside>
      <section className="main">
        <Chat />
      </section>
    </div>
  );
}
