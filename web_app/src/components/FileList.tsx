import cx from "clsx";
import { useState } from "react";
import { Table, ScrollArea, ActionIcon, Text, Group } from "@mantine/core";
import classes from "./FileList.module.css";
import { IconExternalLink, IconUpload, IconX } from "@tabler/icons-react";
import { FilePreviewModal } from "./FilePreviewModal.tsx";
import { FileWithId } from "../App.tsx";
import { API_URL } from "../env.ts";

interface FileListItemProps {
  file: FileWithId;
  onFileClick: (file: File) => void;
  onFileUpload: (file: File) => void;
  onRemoveFile: (id: string) => void;
}

const FileListItem: React.FC<FileListItemProps> = ({
  file,
  onFileClick,
  onFileUpload,
  onRemoveFile,
}) => {
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(false);

  const handleUpload = async () => {
    setUploading(true);
    try {
      await onFileUpload(file.file);
    } finally {
      setUploading(false);
      setUploaded(true);
    }
  };

  return (
    <Table.Tr key={file.id}>
      <Table.Td className={classes.tdName}>
        <Group>
          <Text>{file.file.name}</Text>
          <ActionIcon onClick={() => onFileClick(file.file)} variant="light">
            <IconExternalLink />
          </ActionIcon>
        </Group>
      </Table.Td>
      <Table.Td>
        <Group>
          <ActionIcon
            color="blue"
            variant="filled"
            loading={uploading}
            disabled={uploaded}
            onClick={handleUpload}
          >
            <IconUpload />
          </ActionIcon>
          <ActionIcon
            color="red"
            variant="filled"
            onClick={() => onRemoveFile(file.id)}
          >
            <IconX />
          </ActionIcon>
        </Group>
      </Table.Td>
    </Table.Tr>
  );
};

export interface FileListProps {
  files: FileWithId[];
  setFiles: any;
}

export const FileList: React.FC<FileListProps> = ({ files, setFiles }) => {
  const [scrolled, setScrolled] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewModalOpen, setPreviewModalOpen] = useState(false);

  const handleRemoveFile = (idToRemove: string) => {
    setFiles((prevFiles: FileWithId[]) =>
      prevFiles.filter((file: FileWithId) => file.id !== idToRemove),
    );
  };

  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
    } catch (error) {
      console.error(error);
    }
  };

  const handleFileClick = (file: File) => {
    setSelectedFile(file);
    setPreviewModalOpen(true);
  };

  return (
    <>
      <ScrollArea
        onScrollPositionChange={({ y }) => setScrolled(y !== 0)}
        className={cx(classes.table)}
      >
        <Table>
          <Table.Thead
            className={`${cx(classes.header, { [classes.scrolled]: scrolled })}`}
          >
            <Table.Tr>
              <Table.Th className={classes.thName}>Name</Table.Th>
              <Table.Th className={classes.thRemove}>Actions</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {files.map((file: FileWithId) => (
              <FileListItem
                key={file.id}
                file={file}
                onFileClick={handleFileClick}
                onFileUpload={handleFileUpload}
                onRemoveFile={handleRemoveFile}
              />
            ))}
          </Table.Tbody>
        </Table>
      </ScrollArea>
      {selectedFile && (
        <FilePreviewModal
          file={selectedFile}
          isModalOpen={previewModalOpen}
          setIsModalOpen={setPreviewModalOpen}
        />
      )}
    </>
  );
};
