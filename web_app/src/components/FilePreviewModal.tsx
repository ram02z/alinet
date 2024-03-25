import { Modal, Paper } from "@mantine/core";
import { MIME_TYPES } from "@mantine/dropzone";
import PDFViewer from "./PDFViewer.tsx";

interface FilePreviewModalProps {
  file: File;
  isModalOpen: boolean;
  setIsModalOpen: (isOpen: boolean) => void;
}

export const FilePreviewModal = ({
  file,
  isModalOpen,
  setIsModalOpen,
}: FilePreviewModalProps) => {
  return (
    <Modal
      opened={isModalOpen}
      onClose={() => setIsModalOpen(false)}
      title={file.name}
      size="xl"
    >
      <Paper p="md">
        {file.type === MIME_TYPES.pdf && <PDFViewer file={file} />}
        {file.type === MIME_TYPES.mp4 && (
          <video controls src={URL.createObjectURL(file)} width="100%" />
        )}
      </Paper>
    </Modal>
  );
};
